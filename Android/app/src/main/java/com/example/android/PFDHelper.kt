package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import java.nio.ByteBuffer
import java.nio.FloatBuffer
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
    private var commonUtils: CommonUtils
    private val DIM_BATCH_SIZE = 1
    private val DIM_PIXEL_SIZE = 3

    init {
        val options = TfLiteInitializationOptions.builder()
            .setEnableGpuDelegateSupport(true)
            .build()

        TfLiteVision.initialize(context, options).addOnSuccessListener {
//            objectDetectorListener.onInitialized()
        }.addOnFailureListener {
            // Called if the GPU Delegate is not supported on the device
            TfLiteVision.initialize(context).addOnSuccessListener {
//                objectDetectorListener.onInitialized()
            }.addOnFailureListener {
//                objectDetectorListener.onError("TfLiteVision failed to initialize: "
//                        + it.message)
            }
        }

        // init global objects
        commonUtils = CommonUtils(context)
    }

    // Read ort model into a ByteArray, run in background
    fun readPfdModel(): ByteArray {
        val modelID = R.raw.keypoint_rcnn_op11_quant
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


    fun onnxInference(bitmap: Bitmap, ortSession: OrtSession, env: OrtEnvironment): PfdResult {
        // output object
        val sortedOutput = mutableMapOf<String, Any>()

//        if (bitmap.isRecycled) {
            val inputNameIterator = ortSession.inputNames!!.iterator()
            val inputName0: String = inputNameIterator.next()
            val inputName1: String = inputNameIterator.next()
            val shape = longArrayOf(3, bitmap.height.toLong(), bitmap.width.toLong())

            // resize bitmap
//            val resizedBitmap: Bitmap = commonUtils.resizeBitmap(bitmap, targetImgSize)

            env.use {
                val tensor0 = OnnxTensor.createTensor(env, preProcess(bitmap), shape)

                // TODO: send an empty bitmap to avoid extra computation
                val tensor1 = OnnxTensor.createTensor(env, preProcess(bitmap), shape)

                val temp = tensor0.byteBuffer[0]

                val inputMap = mutableMapOf<String, OnnxTensor>()
                inputMap[inputName0] = tensor0
                inputMap[inputName1] = tensor1

                val output = ortSession?.run(inputMap)

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
//        }

        return if (sortedOutput.isNotEmpty())
            preprocessOutput(sortedOutput)
        else {
            return PfdResult()
        }
    }

    private fun preprocessOutput(modelOutput: MutableMap<String, Any>): PfdResult {
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

        return processedOutput
    }

//    @ExperimentalGetImage
//    fun pfdInference(bitmap: Bitmap): Array<FloatArray> {
//        // declare tensorflow model
//        keypointModel = KeypointModel.newInstance(context)
//
//        val tensorImage = TensorImage(DataType.FLOAT32)
//        tensorImage.load(bitmap)
//
//        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
//            .add(
//                ResizeOp(
//                    targetImgSize,
//                    targetImgSize,
//                    ResizeOp.ResizeMethod.BILINEAR
//                )
//            ) //.add(new Rot90Op(numRotation))
//            .build()
//
//        val processedImg = imageProcessor.process(tensorImage)
//
//        // Creates inputs for reference.
//        val inputFeature0 =
//            TensorBuffer.createFixedSize(
//                intArrayOf(1, targetImgSize, targetImgSize, 3),
//                DataType.FLOAT32
//            )
//        inputFeature0.loadBuffer(processedImg.buffer)
//
//        // Runs model inference and gets result.
//        val outputs = keypointModel.process(inputFeature0)
//        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//        // Create a TensorBuffer for the input tensor
//        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 10, 10, 8), DataType.FLOAT32)
//        outputBuffer.loadBuffer(outputFeature0.buffer)
//
//        // loop through the list of keypoints and draw a circle at each position on the canvas
//        val keypoints =
//            outputBuffer.floatArray // assuming tensorBuffer contains the list of keypoints
//
//        val x = 100
//        val y = 4
//        val z = 2
//        val imgSize = 512
//
//        val reshapedArray = Array(x) {
//            Array(y) {
//                FloatArray(z)
//            }
//        }
//
//        var i = 0
//        for (b in 0 until x) {
//            for (h in 0 until y) {
//                for (w in 0 until z) {
//                    reshapedArray[b][h][w] = keypoints[i++] * imgSize
//                }
//            }
//        }
//
//        return reshapedArray[0]
//    }

    fun saveVideoFromBitmaps(
        frames: List<Bitmap>,
        outputPath: String,
        width: Int,
        height: Int,
        frameRate: Int
    ) {
        val mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val mediaFormat = MediaFormat.createVideoFormat("video/avc", width, height)
        mediaFormat.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface
        )
        mediaFormat.setInteger(MediaFormat.KEY_BIT_RATE, 2000000)
        mediaFormat.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
        mediaFormat.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 5)
        val trackIndex = mediaMuxer.addTrack(mediaFormat)
        mediaMuxer.start()

        val bufferInfo = MediaCodec.BufferInfo()
        val encoder = MediaCodec.createEncoderByType("video/avc")
        encoder.configure(mediaFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        val inputSurface = encoder.createInputSurface()
        inputSurface?.let {
            encoder.start()
            val canvas = inputSurface.lockCanvas(null)
            frames.forEach { bitmap ->
                canvas.drawBitmap(bitmap, 0f, 0f, null)
                inputSurface?.let {
                    val inputBufferId = encoder.dequeueInputBuffer(-1)
                    if (inputBufferId >= 0) {
                        val inputBuffer = encoder.getInputBuffer(inputBufferId)
                        inputBuffer?.let {
                            it.clear()
                            val byteBuffer = ByteBuffer.allocate(bitmap.byteCount)
                            bitmap.copyPixelsToBuffer(byteBuffer)
                            inputBuffer.put(byteBuffer)
                            encoder.queueInputBuffer(inputBufferId, 0, byteBuffer.capacity(), 0, 0)
                        }
                    }
                    var outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 0)
                    while (outputBufferId >= 0) {
                        val outputBuffer = encoder.getOutputBuffer(outputBufferId)
                        outputBuffer?.let {
                            mediaMuxer.writeSampleData(trackIndex, outputBuffer, bufferInfo)
                        }
                        encoder.releaseOutputBuffer(outputBufferId, false)
                        outputBufferId = encoder.dequeueOutputBuffer(bufferInfo, 0)
                    }
                }
            }
            inputSurface.unlockCanvasAndPost(canvas)
            encoder.stop()
            encoder.release()
        }
        mediaMuxer.stop()
        mediaMuxer.release()
    }

    fun perspectiveTransformation(imageBitmap: Bitmap, keyPoints: MutableList<FloatArray>): Bitmap {
        val imgMat = commonUtils.bitmapToMat(imageBitmap)

        // Convert points array to OpenCV Point array
        // TODO: uncomment if keypoint isn't a test point
//        var topLeft = floatArrayOf(0f, 0f)
//        var topRight = floatArrayOf(0f, 0f)
//        var bottomLeft = floatArrayOf(0f, 0f)
//        var bottomRight = floatArrayOf(0f, 0f)
//
//        val bboxCenter: LocalUtils.BboxCenter = localUtils.calculateBboxCenter(keyPoints)
//
//        for (kp in keyPoints) {
//            if (kp[0] < bboxCenter.centerX && kp[1] < bboxCenter.centerY) {
//                topLeft = kp
//            } else if (kp[0] > bboxCenter.centerX && kp[1] > bboxCenter.centerY) {
//                bottomRight = kp
//            } else if (kp[0] < bboxCenter.centerX && kp[1] > bboxCenter.centerY) {
//                topRight = kp
//            } else if (kp[0] > bboxCenter.centerX && kp[1] < bboxCenter.centerY) {
//                bottomLeft = kp
//            }
//        }

        val topLeft = keyPoints[0]
        val topRight = keyPoints[3]
        val bottomLeft = keyPoints[1]
        val bottomRight = keyPoints[2]

        val imgWidth = abs(topLeft[0] - topRight[0])
        val imgHeight = abs(topLeft[1] - bottomLeft[1])

        // srcPoints => (top_left, top_right, bottom_left, bottom_right)
        val srcPoints = arrayOf(
            CVPoint(topRight[0].toDouble(), topRight[1].toDouble()),
            CVPoint(topLeft[0].toDouble(), topLeft[1].toDouble()),
            CVPoint(bottomLeft[0].toDouble(), bottomLeft[1].toDouble()),
            CVPoint(bottomRight[0].toDouble(), bottomRight[1].toDouble())
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