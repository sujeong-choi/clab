package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
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
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import org.opencv.core.Point as CVPoint


class KeypointListType(val value: MutableList<MutableList<FloatArray>>)

class BboxListType(val value: MutableList<FloatArray>)

data class PfdResult(
    var keypoint: KeypointListType = KeypointListType(mutableListOf()),
    var bbox: BboxListType = BboxListType(mutableListOf()),
    var size: Int = 0
) {}

class PFDHelper(val context: Context) {
    private var commonUtils: CommonUtils = CommonUtils(context)
    private val DIM_BATCH_SIZE = 1
    private val DIM_PIXEL_SIZE = 3
    private val targetSize: Int = 512

    // Read ort model into a ByteArray, run in background
    fun readPfdModel(): ByteArray {
        val modelID = R.raw.keypoint_rcnn
        return context.resources.openRawResource(modelID).readBytes()
    }

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


    fun pfdInference(bitmap: Bitmap, ortSession: OrtSession): PfdResult {
        // output object
        val sortedOutput = mutableMapOf<String, Any>()

        // resize bitmap
        val resizedBitmap: Bitmap = resizeBitmap(bitmap, targetSize)

        val inputNameIterator = ortSession.inputNames!!.iterator()
        val inputName0: String = inputNameIterator.next()
        val inputName1: String = inputNameIterator.next()
        val shape = longArrayOf(3, resizedBitmap.height.toLong(), resizedBitmap.width.toLong())


        val env = GlobalVars.ortEnv
        env.use {
            val tensor0 = OnnxTensor.createTensor(env, preProcess(resizedBitmap), shape)

            // send an empty bitmap to avoid extra computation
            val bufferSize = shape.reduce(Long::times).toInt()
            val buffer = FloatBuffer.allocate(bufferSize)
            val tensor1 = OnnxTensor.createTensor(env, buffer, shape)
            val inputMap = mutableMapOf<String, OnnxTensor>()

            inputMap[inputName0] = tensor0
            inputMap[inputName1] = tensor1

            val output = ortSession.run(inputMap)

            val outputNames = ortSession.outputNames.toList()

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

        return if (sortedOutput.isNotEmpty())
            preprocessOutput(sortedOutput, bitmap.width, bitmap.height)
        else {
            return PfdResult()
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
            if (scores[idx] >= 0.8) {
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

        return resizeOutput(processedOutput, origWidth, origHeight)
    }

    private fun resizeOutput(
        pfdResult: PfdResult,
        origWidth: Int,
        origHeight: Int
    ): PfdResult {
        val resizedOutput = PfdResult()

        // calculate width and height ratio
        val widthRatio = origWidth.toFloat() / targetSize.toFloat()
        val heightRatio = origHeight.toFloat() / targetSize.toFloat()

        var bboxes = pfdResult.bbox.value
        var keypointsList = pfdResult.keypoint.value


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

    private fun resizeBitmap(bitmap: Bitmap, targetImgSize: Int): Bitmap {
        // resize bitmap
        return Bitmap.createScaledBitmap(bitmap, targetImgSize, targetImgSize, false)
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

    private fun isPointInsideRectangle(point: FloatArray, bbox: FloatArray): Boolean {
        val topLeftX = bbox[0] - 10
        val topLeftY = bbox[1] - 10
        val bottomRightX = bbox[2] + 10
        val bottomRightY = bbox[3] + 10

        val pointX = point[0]
        val pointY = point[1]

        return pointX in topLeftX..bottomRightX && pointY in topLeftY..bottomRightY
    }

    fun saveVideoFromBitmaps(
        frames: List<Bitmap>
    ) {
        val file = File(getVideoFilePath("timelapse"))
        file.createNewFile()

        var out: FileChannelWrapper? = null
        try {
            out = NIOUtils.writableFileChannel(file.absolutePath)
            val enc =
                AndroidSequenceEncoder.createSequenceEncoder(file, 2)

            for (frame in frames) {
                enc.encodeImage(frame)
            }

            enc.finish()
        } catch (e: Exception) {
            e.message?.let { Log.v("PFD", it) }
        }
    }

    private fun getVideoFilePath(postfix: String = ""): String {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        val currentDateAndTime: String = dateFormat.format(Date())
        val fileName = "$currentDateAndTime$postfix.mp4"
        val fileDir = context.filesDir
        return "$fileDir/$fileName"
    }


    fun perspectiveTransformation(imageBitmap: Bitmap, keyPoints: MutableList<FloatArray>): Bitmap {
        val imgMat = commonUtils.bitmapToMat(imageBitmap)

        val topLeft = keyPoints[0]
        val topRight = keyPoints[3]
        val bottomLeft = keyPoints[1]
        val bottomRight = keyPoints[2]

        val imgWidth = abs(topLeft[0] - topRight[0])
        val imgHeight = abs(topLeft[1] - bottomLeft[1])

        // srcPoints => (top_left, top_right, bottom_left, bottom_right)
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

    fun perspectiveTransformation(
        imageBitmapList: List<Bitmap>,
        keyPoint: MutableList<FloatArray>
    ): List<Bitmap> {

        var transformedList: List<Bitmap> = mutableListOf<Bitmap>()

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