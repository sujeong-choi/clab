package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mediapipe.formats.proto.LandmarkProto
import org.jcodec.api.android.AndroidSequenceEncoder
import org.jcodec.common.io.FileChannelWrapper
import org.jcodec.common.io.NIOUtils
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.max
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

    private val visibilityThreshold = 0.75

    /**
     * Performs object detection inference on a given bitmap image using the ONNX runtime.
     * @param bitmap the bitmap image to perform inference on.
     * @param pfdSession the ONNX session for performing inference.
     * @return PfdResult object containing the output of the inference, including bounding box coordinates,
     * one-hot encoded labels, scores for each bounding box, and keypoint information.
     * @throws OrtException if there is an issue running the inference.
     */
    fun pfdInference(bitmap: Bitmap, pfdSession: OrtSession?): PfdResult {
        // output object
        val sortedOutput = mutableMapOf<String, Any>()

        // resize bitmap for memory optimization, disabled to avoid model accuracy
        val resizedBitmap: Bitmap = commonUtils.resizeBitmap(bitmap, targetSize)

        val inputNameIterator = pfdSession?.inputNames!!.iterator()
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

            } catch (e: OrtException) {
                e.message?.let { it1 -> Log.v("PFD", it1) }
            }

            return if (sortedOutput.isNotEmpty()) {
                preprocessOutput(sortedOutput)
            }
            else {
                return PfdResult()
            }
        }
    }

    /**
     * Preprocesses a given bitmap for use as an input tensor in the PFD model.
     * The bitmap is resized and converted to a float buffer containing pixel values
     * normalized to the range [0, 1].
     * @param bitmap The bitmap to preprocess.
     * @return A float buffer containing the preprocessed pixel data.
     */
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

    /**
     * Processes the output of the PFD model by filtering the keypoints and bounding boxes based on a score threshold.
     * @param modelOutput the output of the PFD model as a MutableMap<String, Any> object.
     * @param pfdThreshold the threshold for filtering the keypoints and bounding boxes based on their scores. Default value is 0.75
     * @return PfdResult object that contains the processed keypoints and bounding boxes.
     */
    private fun preprocessOutput(
        modelOutput: MutableMap<String, Any>,
        pfdThreshold: Double = 0.75
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
            if (scores[idx] >= pfdThreshold) {
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

        // resize keypoints from 512x512 to mediaPipeSize
        if(processedOutput.keypoint.value.size > 0) {
            val resizedKeypoints = resizeKeypointsToMediapipe(processedOutput.keypoint.value[0])
            processedOutput.keypoint.value[0] = resizedKeypoints
        }

        return processedOutput
    }

    /**
     * Resizes the keypoints to the size compatible with the Mediapipe model.
     * @param keypoints the original list of keypoints to be resized
     * @return the resized keypoints as a MutableList<FloatArray>
     */
    private fun resizeKeypointsToMediapipe(
        keypoints: MutableList<FloatArray>
    ): MutableList<FloatArray> {
        val resizedOutput: MutableList<FloatArray> = mutableListOf()

        // custom image size
        val widthRatio = GlobalVars.targetMediapipeRes.width / targetSize.toFloat()
        val heightRatio = GlobalVars.targetMediapipeRes.height / targetSize.toFloat()

        for (kp in keypoints) {
            resizedOutput.add(
                floatArrayOf(
                    kp[0] * widthRatio,
                    kp[1] * heightRatio

                )
            )
        }

        return resizedOutput
    }

    /**
     * Resizes the keypoints to the target size of the Mediapipe model.
     * @param keypoints the list of keypoints to be resized.
     * @return a list of resized keypoints.
     */
    private fun resizeBboxToMediapipe(
        bbox: FloatArray
    ): FloatArray {

        // custom image size
        val widthRatio = GlobalVars.targetMediapipeRes.width / targetSize.toFloat()
        val heightRatio = GlobalVars.targetMediapipeRes.height / targetSize.toFloat()


        // resize keypoints
        var resizedOutput: FloatArray = floatArrayOf(
            (bbox[0] * widthRatio).toFloat(),
            (bbox[1] * heightRatio).toFloat(),
            (bbox[2] * widthRatio).toFloat(),
            (bbox[3] * heightRatio).toFloat()
        )

        return resizedOutput
    }

    /**
     * Determines whether a hand is in the frame of a given input frame by checking the location of its keypoints
     * relative to the detected human pose landmarks. If isStrict is set to true, it enforces the presence
     * of a human skeleton in the picture to test if the hand is in the frame or not.
     * @param inputFrame The input bitmap frame to check for a hand.
     * @param keypoint A list of keypoint coordinates for the hand.
     * @param landMarks The list of human pose landmarks detected in the frame.
     * @param isStrict If true, it enforces human skeleton presence in picture to test if hand is in frame or not.
     * @return true if the hand is in the frame; false otherwise.
     */
    fun isHandInFrame(
        inputFrame: Bitmap,
        keypoint: MutableList<FloatArray>,
        landMarks: LandmarkProto.NormalizedLandmarkList?,
        isStrict: Boolean = false
    ): Boolean {
        if (landMarks != null) {
            val filteredLandmarks = landMarks.landmarkList.toList().slice(1..22)

            // only use x and y coordinates from filteredLandmarks list
            // convert coordinates to the correct aspect ratio
            val poseLandmarks: MutableList<FloatArray> = mutableListOf()

            for (landmark in filteredLandmarks) {
                if (landmark.visibility < visibilityThreshold)
                    poseLandmarks.add(
                        floatArrayOf(
                            0f,
                            0f
                        )
                    )
                else if (landmark.visibility >= visibilityThreshold)
                    poseLandmarks.add(
                        floatArrayOf(
                            (landmark.x * GlobalVars.targetMediapipeRes.width).toFloat(),
                            (landmark.y * GlobalVars.targetMediapipeRes.height).toFloat()
                        )
                    )
            }

            // get max and min x and y values
            val minX = keypoint.minByOrNull { it[0] }?.get(0) ?: 0.0f
            val minY = keypoint.minByOrNull { it[1] }?.get(1) ?: 0.0f
            val maxX = keypoint.maxByOrNull { it[0] }?.get(0) ?: 0.0f
            val maxY = keypoint.maxByOrNull { it[1] }?.get(1) ?: 0.0f

            for (pose in poseLandmarks) {
                // check if point is inside or outside the bbox
                if (isPointInsideRectangle(pose, floatArrayOf(minX, minY, maxX, maxY))) {
                    return true
                }
            }
            return false
        }
        // if isStrict is set to true, it enforces human skeleton presence in picture to test if hand is in frame or not
        return isStrict
    }

    private fun isPointInsideRectangle(
        point: FloatArray,
        keypoints: FloatArray,
        bufferZone: Int = 0
    ): Boolean {

        val topLeftX = keypoints[0] - bufferZone
        val topLeftY = keypoints[1] - bufferZone
        val bottomRightX = keypoints[2] + bufferZone
        val bottomRightY = keypoints[3] + bufferZone

        val pointX = point[0]
        val pointY = point[1]

        return pointX in topLeftX..bottomRightX && pointY in topLeftY..bottomRightY
    }

    /**
     * Compares two input images to check if they are different by computing the mean square error (MSE) between the grayscale
     * versions of the images. If the MSE is greater than the specified threshold, the function returns true, indicating
     * that the images are different. Otherwise, the function returns false.
     * @param inputFrame The current input image to compare
     * @param prevInputFrame The previous input image to compare
     * @param threshold The threshold value for the MSE above which the images are considered different. Default is 0.0.
     * @return Returns true if the two images are different, false otherwise.
     */
    fun compareImages(inputFrame: Bitmap, prevInputFrame: Bitmap, threshold: Double): Boolean {
        val img1 = Mat()
        val img2 = Mat()

        Utils.bitmapToMat(inputFrame, img1)
        Utils.bitmapToMat(prevInputFrame, img2)

        // Convert images to grayscale
        val gray1 = Mat()
        val gray2 = Mat()
        Imgproc.cvtColor(img1, gray1, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(img2, gray2, Imgproc.COLOR_BGR2GRAY);

        // Calculate mean square error between the images
        val diff = Mat()
        Core.absdiff(gray1, gray2, diff)
        val squaredDiff = Mat()
        Core.multiply(diff, diff, squaredDiff)
        val mse = MatOfDouble()
        Core.mean(squaredDiff, mse)

        // Compare mean square error to threshold
        val mseValue = mse.toArray()[0]
        return mseValue > threshold
    }


    /**
     * Saves a video from a list of bitmaps to internal storage as an mp4 file using the given fps.
     * The video is saved under the "timelapse" folder with a filename of "timelapse.mp4".
     * @param fps the frames per second for the output video
     * @param timeLapseId the id of the timelapse to save
     * @param totalFrames the total number of frames in the timelapse
     * @return true if the video was successfully saved, false otherwise
     */
    fun saveVideoFromBitmaps(
        fps: Int,
        timeLapseId: String,
        totalFrames: Int
    ): Boolean {

        val frameList: Array<Bitmap> =
            retrieveAndDeleteBitmapsFromInternalStorage(timeLapseId, totalFrames)

        if (frameList.size > 30) {
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

    /**
     * This function gets the id of the timelapse based on current time and date
     * @return a string containing a unique timelapse id to be used for filename
     */
    fun getTimelapseId(): String {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        return dateFormat.format(Date())
    }

    /**
     * This function saves a bitmap to internal storage with a specific ID and count.
     * It returns the file path of the saved bitmap.
     * @param bitmap The bitmap to save.
     * @param id The specific ID to use in the file name.
     * @param count The count to use in the file name.
     * @return a string of the saved bitmap
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
     * @param id The specific ID to use in the file name.
     * @return An array of Bitmap objects.
     */
    private fun retrieveAndDeleteBitmapsFromInternalStorage(
        id: String,
        totalFrames: Int
    ): Array<Bitmap> {
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

    /**
     * Applies perspective transformation to the input image based on the given key points.
     * @param imageBitmap The input image bitmap.
     * @param keyPoints A list of key points that define the region of interest in the input image.
     * The first key point should correspond to the top-left corner of the ROI, and the subsequent
     * points should be specified in clockwise order.
     * @return The transformed image bitmap.
     */
    fun perspectiveTransformation(imageBitmap: Bitmap, keyPoints: MutableList<FloatArray>): Bitmap {
        val imgMat = commonUtils.bitmapToMat(imageBitmap)

        val topLeft = keyPoints[0]
        val topRight = keyPoints[3]
        val bottomLeft = keyPoints[1]
        val bottomRight = keyPoints[2]

        val imgWidth = max(abs(topLeft[0] - topRight[0]), abs(bottomLeft[0]-bottomRight[0]))
        val imgHeight = max(abs(topLeft[1] - bottomLeft[1]),abs(topRight[1]-bottomRight[1]))

        val srcPoints = arrayOf(
            CVPoint(topLeft[0].toDouble(), (topLeft[1]).toDouble()),
            CVPoint(topRight[0].toDouble(), (topRight[1]).toDouble()),
            CVPoint(bottomRight[0].toDouble(), (bottomRight[1]).toDouble()),
            CVPoint(bottomLeft[0].toDouble(), (bottomLeft[1]).toDouble())
        )
        val dstPoints = arrayOf(
            CVPoint(0.0, 0.0),
            CVPoint(imgWidth.toDouble() - 1, 0.0),
            CVPoint(imgWidth.toDouble() - 1, imgHeight.toDouble() - 1),
            CVPoint(0.0, imgHeight.toDouble() - 1)
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

    /**
     * Applies perspective transformation on a list of input bitmaps using the provided key points.
     * @param imageBitmapList List of bitmaps to apply the transformation on.
     * @param keyPoint The key points to use for the transformation.
     * @return The list of transformed bitmaps.
     */
    fun perspectiveTransformation(
        imageBitmapList: List<Bitmap>,
        keyPoint: MutableList<FloatArray>,
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