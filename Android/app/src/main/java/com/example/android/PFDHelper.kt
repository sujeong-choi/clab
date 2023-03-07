package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mediapipe.formats.proto.LandmarkProto
import org.jcodec.api.android.AndroidSequenceEncoder
import org.jcodec.common.io.FileChannelWrapper
import org.jcodec.common.io.NIOUtils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.thread
import kotlin.math.abs
import org.opencv.core.Point as CVPoint


// Keypoint and bbox datatypes
class KeypointListType(val value: MutableList<MutableList<FloatArray>>)

class BboxListType(val value: MutableList<FloatArray>)

data class PfdResult(
    var keypoint: KeypointListType = KeypointListType(mutableListOf()),
    var bbox: BboxListType = BboxListType(mutableListOf()),
    var size: Int = 0
) {}


/**
 * PFD Helper class containing logic related to PFD Prediction and Detection and Timelapse Sampling
 */
class PFDHelper(val context: Context) {
    private var commonUtils: CommonUtils = CommonUtils(context)
    private val DIM_BATCH_SIZE = 1
    private val DIM_PIXEL_SIZE = 3
    private val targetSize: Int = 512

    fun preProcess(bitmap: Bitmap): FloatBuffer {
        val imgWidth = bitmap.width
        val imgHeight = bitmap.height

        val imgData = FloatBuffer.allocate(
            DIM_BATCH_SIZE
                    * DIM_PIXEL_SIZE
                    * imgWidth
                    * imgHeight
        )
        imgData.rewind()
        val stride = imgWidth * imgHeight
        val bmpData = IntArray(stride)
        bitmap.getPixels(bmpData, 0, imgWidth, 0, 0, imgWidth, imgHeight)
        for (i in 0..imgWidth - 1) {
            for (j in 0..imgHeight - 1) {
                val idx = imgHeight * i + j
                val pixelValue = bmpData[idx]
                imgData.put(idx, ((pixelValue shr 16 and 0xFF) / 255f))
                imgData.put(idx + stride, ((pixelValue shr 8 and 0xFF) / 255f))
                imgData.put(idx + stride * 2, ((pixelValue and 0xFF) / 255f))
            }
        }

        imgData.rewind()
        return imgData
    }


    fun pfdInference(bitmap: Bitmap, pfdSession: OrtSession?): PfdResult {
        // output object
        val sortedOutput = mutableMapOf<String, Any>()

        // resize bitmap for memory optimization, disabled to avoid model accuracy
//        val resizedBitmap: Bitmap = resizeBitmap(bitmap, targetSize)

        val inputNameIterator = pfdSession?.inputNames!!.iterator()
        val inputName0: String = inputNameIterator.next()
        val inputName1: String = inputNameIterator.next()
        val shape = longArrayOf(3, bitmap.height.toLong(), bitmap.width.toLong())


        val env = GlobalVars.ortEnv
        env.use {
            val tensor0 = OnnxTensor.createTensor(env, preProcess(bitmap), shape)

            // send an empty bitmap to avoid extra computation
            val bufferSize = shape.reduce(Long::times).toInt()
            val buffer = FloatBuffer.allocate(bufferSize)
            val tensor1 = OnnxTensor.createTensor(env, buffer, shape)
            val inputMap = mutableMapOf<String, OnnxTensor>()

            inputMap[inputName0] = tensor0
            inputMap[inputName1] = tensor1

            try {

                val output = pfdSession.run(inputMap)

                val outputNames = pfdSession.outputNames.toList()

                /*
                * Output format from the model is as follows
                * boxes_1: float coordinates containing coordinates of bounding boxes in the format [[top_left_x, top_left_y, bottom_right_x, bottom_right_y]]
                * labels_1: one hot encoding of labels
                * scores_1: scores of each bounding box
                * keypoints_1: list of 4 keypoints for each bounding box, each keypoint contains [x, y, a] values
                * keypoint_scores_1: scores of each keypoints
                * */
                for (idx in outputNames.indices) {
                    val name = outputNames[idx]
                    sortedOutput[name] = output?.get(idx)?.value as Any

                    // break on the 5th item since the second image is a dummy image
                    if (idx == 4) break
                }
            }
            catch (e: Exception) {
                e.message?.let { it1 -> Log.v("PFD", it1) }
            }

            return if (sortedOutput.isNotEmpty())
                preprocessOutput(sortedOutput, bitmap.width, bitmap.height)
            else {
                return PfdResult()
            }
        }
    }

    private fun preprocessOutput(
        modelOutput: MutableMap<String, Any>,
        origWidth: Int,
        origHeight: Int
    ): PfdResult {
        // check if scores for both bbox and keypoints
        // filter them by score > 0.8
        val processedOutput = PfdResult()

        // filter keypoints and bboxes
        val scores = modelOutput["scores_1"] as FloatArray
        val bboxes = modelOutput["boxes_1"] as Array<FloatArray>
        val keypoints = modelOutput["keypoints_1"] as Array<Array<FloatArray>>
        var size = 0

        for (idx in scores.indices) {
            if (scores[idx] >= 0.7) {
                // add boxes to processed output
                val bbox: FloatArray = bboxes[idx]
                processedOutput.bbox.value.add(bbox)

                // add keypoints to processed output
                val keypoint: MutableList<FloatArray> = keypoints[idx].toMutableList()
                processedOutput.keypoint.value.add(keypoint)

                // size
                size++
            }
        }

        processedOutput.size = size

        // to be used when input image is resized, output keypoints also need to be resized
        // return resizeOutput(processedOutput, origWidth, origHeight)

        return processedOutput
    }

    // resize keypoint outputs
    private fun resizeOutput(
        pfdResult: PfdResult,
        origWidth: Int,
        origHeight: Int
    ): PfdResult {
        val resizedOutput = PfdResult()

        // calculate width and height ratio
        val widthRatio = origWidth.toFloat() / targetSize.toFloat()
        val heightRatio = origHeight.toFloat() / targetSize.toFloat()

        val bboxes = pfdResult.bbox.value
        val keypointsList = pfdResult.keypoint.value


        // resize bbox and keypoint
        for (i in bboxes.indices) {
            val bbox = bboxes[i]
            val keypoints = keypointsList[i]

            resizedOutput.bbox.value.add(
                floatArrayOf(
                    bbox[0] * widthRatio,
                    bbox[1] * heightRatio,
                    bbox[2] * widthRatio,
                    bbox[3] * heightRatio
                )
            )

            val tempKp: MutableList<FloatArray> = mutableListOf()

            for (kp in keypoints) {
                tempKp.add(
                    floatArrayOf(
                        kp[0] * widthRatio,
                        kp[1] * heightRatio
                    )
                )
            }
            resizedOutput.keypoint.value.add(tempKp)
        }

        resizedOutput.size = pfdResult.size

        return resizedOutput
    }

    fun isHandInFrame(
        inputFrame: Bitmap,
        bbox: FloatArray,
        landMarks: LandmarkProto.NormalizedLandmarkList?
    ): Boolean {
        if (landMarks != null) {
            val filteredLandmarks = landMarks.landmarkList.toList().slice(13..22)

            val frameWidth = inputFrame.width
            val frameHeight = inputFrame.height

            // only use x and y coordinates from filteredLandmarks list
            // convert coordinates to the correct aspect ratio
            val poseLandmarks: MutableList<FloatArray> = mutableListOf()

            for (landmark in filteredLandmarks) {
                if (landmark.visibility < 0.5)
                    poseLandmarks.add(
                        floatArrayOf(
                            0f,
                            0f
                        )
                    )
                else if (landmark.visibility >= 0.5)
                    poseLandmarks.add(
                        floatArrayOf(
                            landmark.x * frameWidth,
                            landmark.y * frameHeight
                        )
                    )
            }

            for (pose in poseLandmarks) {
                // check if point is inside or outside the bbox
                if (isPointInsideRectangle(pose, bbox)) {
                    return true
                }
            }
            return false
        }
        return false
    }

    private fun isPointInsideRectangle(
        point: FloatArray,
        bbox: FloatArray,
        bufferZone: Int = 10
    ): Boolean {
        val topLeftX = bbox[0] - bufferZone
        val topLeftY = bbox[1] - bufferZone
        val bottomRightX = bbox[2] + bufferZone
        val bottomRightY = bbox[3] + bufferZone

        val pointX = point[0]
        val pointY = point[1]

        return pointX in topLeftX..bottomRightX && pointY in topLeftY..bottomRightY
    }

    fun saveVideoFromBitmaps(
        fps: Int,
        timeLapseId: String,
        totalFrames: Int
    ): Boolean {

        val frameList: Array<Bitmap> = retrieveAndDeleteBitmapsFromInternalStorage(timeLapseId, totalFrames)

        if(frameList.size > 30) {
            val file = File(getVideoFilePath("timelapse"))
            var out: FileChannelWrapper? = null
            var isCompleted = false

            val tlWidth =
                if (frameList[0].width % 2 == 0) frameList[0].width else frameList[0].width - 1
            val tlHeight =
                if (frameList[0].height % 2 == 0) frameList[0].height else frameList[0].height - 1

            val timelapseSaveThread = thread {
                try {
                    out = NIOUtils.writableFileChannel(file.absolutePath)
                    val enc =
                        AndroidSequenceEncoder.createSequenceEncoder(file, fps)

                    for (frame in frameList) {
                        val resizedBitmap =
                            Bitmap.createScaledBitmap(frame, tlWidth, tlHeight, false)
                        enc.encodeImage(resizedBitmap)
                    }

                    enc.finish()

                    // recycle all timelapse frames to freeup resources
                    frameList.map { bitmap: Bitmap -> bitmap.recycle() }

                    isCompleted = true
                } catch (e: Exception) {
                    e.message?.let { Log.v("PFD", it) }
                } finally {
                    NIOUtils.closeQuietly(out)
                }
            }

            timelapseSaveThread.join()
            return isCompleted
        }
        return false
    }

    private fun getVideoFilePath(postfix: String = ""): String {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        val currentDateAndTime: String = dateFormat.format(Date())
        val fileName = "$currentDateAndTime$postfix.mp4"
        val fileDir = context.filesDir
        return "$fileDir/$fileName"
    }

    fun getTimelapseId(): String {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        return dateFormat.format(Date())
    }

    /**
     * This function saves a bitmap to internal storage with a specific ID and count.
     * It returns the file path of the saved bitmap.
     * @param context The context of the application.
     * @param bitmap The bitmap to save.
     * @param id The specific ID to use in the file name.
     * @param count The count to use in the file name.
     */
    fun saveBitmapToInternalStorage(bitmap: Bitmap, id: String, count: Int): String {
        val filename = "${id}_$count.png"
        val file = File(context.filesDir, filename)
        val stream = FileOutputStream(file)

        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
        stream.flush()
        stream.close()

        return file.absolutePath
    }

    /**
     * This function retrieves all bitmaps from internal storage with a specific ID and
     * puts them into an array. It then deletes the bitmaps from internal storage.
     * @param context The context of the application.
     * @param id The specific ID to use in the file name.
     * @return An array of Bitmap objects.
     */
    private fun retrieveAndDeleteBitmapsFromInternalStorage(id: String, totalFrames: Int): Array<Bitmap> {
        val bitmapList = mutableListOf<Bitmap>()
        var count = 1

        while (count <= totalFrames) {
            val filename = "${id}_$count.png"
            val file = File(context.filesDir, filename)

            if (!file.exists()) {
                break
            }

            val stream = FileInputStream(file)
            val bitmap = BitmapFactory.decodeStream(stream)
            stream.close()

            file.delete()

            bitmapList.add(bitmap)
            count++
        }

        return bitmapList.toTypedArray()
    }

    fun perspectiveTransformation(imageBitmap: Bitmap, keyPoints: MutableList<FloatArray>): Bitmap {
        val imgMat = commonUtils.bitmapToMat(imageBitmap)

        val topLeft = keyPoints[0]
        val topRight = keyPoints[3]
        val bottomLeft = keyPoints[1]
        val bottomRight = keyPoints[2]

        val imgWidth = abs(topLeft[0] - topRight[0])
        val imgHeight = abs(topLeft[1] - bottomLeft[1])

        val srcPoints = arrayOf(
            CVPoint(topLeft[0].toDouble(), topLeft[1].toDouble()),
            CVPoint(topRight[0].toDouble(), topRight[1].toDouble()),
            CVPoint(bottomRight[0].toDouble(), bottomRight[1].toDouble()),
            CVPoint(bottomLeft[0].toDouble(), bottomLeft[1].toDouble())
        )
        val dstPoints = arrayOf(
            CVPoint(0.0, 0.0),
            CVPoint(imgWidth.toDouble(), 0.0),
            CVPoint(imgWidth.toDouble(), imgHeight.toDouble()),
            CVPoint(0.0, imgHeight.toDouble())
        )

        // Perform perspective transformation
        val srcMat = MatOfPoint2f(*srcPoints)
        val dstMat = MatOfPoint2f(*dstPoints)
        val perspectiveTransform = Imgproc.getPerspectiveTransform(srcMat, dstMat)
        val transformedMat = Mat()
        Imgproc.warpPerspective(
            imgMat,
            transformedMat,
            perspectiveTransform,
            Size(imgWidth.toDouble(), imgHeight.toDouble())
        )

        // Convert back to ImageProxy format
        return commonUtils.matToBitmap(transformedMat)!!
    }

    // perspective transformation for a list of bitmaps instead of one
    fun perspectiveTransformation(
        imageBitmapList: List<Bitmap>,
        keyPoint: MutableList<FloatArray>
    ): List<Bitmap> {

        val transformedList: List<Bitmap> = mutableListOf<Bitmap>()

        for (i in imageBitmapList.indices) {
            val transformedBitmap = perspectiveTransformation(imageBitmapList[i], keyPoint)
            transformedList.plus(transformedBitmap)
        }

        return transformedList
    }

    fun destroyModel() {
        // Releases model resources if no longer used.
    }
}