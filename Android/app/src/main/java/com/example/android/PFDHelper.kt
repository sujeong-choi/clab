package com.example.android

import android.content.Context
import android.graphics.*
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import androidx.camera.core.ExperimentalGetImage
import com.example.android.ml.KeypointModel
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CVPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import org.tensorflow.lite.task.gms.vision.segmenter.ImageSegmenter
import java.nio.ByteBuffer
import kotlin.math.abs


class PFDHelper(val context: Context) {
    private lateinit var keypointModel: KeypointModel
    private var numThreads: Int = 2
    private var currentDelegate: Int = 0
    private var imageSegmenter: ImageSegmenter? = null
    private var localUtils: LocalUtils

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
        localUtils = LocalUtils(context)
    }


    @ExperimentalGetImage
    fun pfdInference(bitmap: Bitmap): Array<FloatArray> {
        // declare tensorflow model
        keypointModel = KeypointModel.newInstance(context)

        // target image size to feed to model
        val targetImgSize = 512

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    targetImgSize,
                    targetImgSize,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            ) //.add(new Rot90Op(numRotation))
            .build()

        val processedImg = imageProcessor.process(tensorImage)

        val keypointModel = KeypointModel.newInstance(context)

        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 512, 512, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImg.buffer)

        // Runs model inference and gets result.
        val outputs = keypointModel.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Create a TensorBuffer for the input tensor
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 10, 10, 8), DataType.FLOAT32)
        outputBuffer.loadBuffer(outputFeature0.buffer)

        // loop through the list of keypoints and draw a circle at each position on the canvas
        val keypoints =
            outputBuffer.floatArray // assuming tensorBuffer contains the list of keypoints

        val x = 100
        val y = 4
        val z = 2
        val imgSize = 512

        val reshapedArray = Array(x) {
            Array(y) {
                FloatArray(z)
            }
        }

        var i = 0
        for (b in 0 until x) {
            for (h in 0 until y) {
                for (w in 0 until z) {
                    reshapedArray[b][h][w] = keypoints[i++] * imgSize
                }
            }
        }

        return reshapedArray[0]
    }

    fun saveVideoFromBitmaps(frames: List<Bitmap>, outputPath: String, width: Int, height: Int, frameRate: Int) {
        val mediaMuxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val mediaFormat = MediaFormat.createVideoFormat("video/avc", width, height)
        mediaFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
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

    fun perspectiveTransformation(imageBitmap: Bitmap, keyPoints: Array<FloatArray>): Bitmap {
        val imgMat = localUtils.bitmapToMat(imageBitmap)

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

        var topLeft = keyPoints[0]
        var topRight = keyPoints[1]
        var bottomLeft = keyPoints[3]
        var bottomRight = keyPoints[2]

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
        return localUtils.matToBitmap(transformedMat)!!
    }

    fun perspectiveTransformation(imageBitmapList: List<Bitmap>, keyPoint: Array<FloatArray>): List<Bitmap> {

        var transformedList: List<Bitmap> = mutableListOf<Bitmap>()

        for (i in imageBitmapList.indices) {
            val transformedBitmap = perspectiveTransformation(imageBitmapList[i], keyPoint)
            transformedList.plus(transformedBitmap)
        }

        return transformedList
    }

    fun destroyModel() {
        // Releases model resources if no longer used.
        keypointModel.close()
    }
}