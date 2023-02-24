package com.example.android

import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.media.Image
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.example.android.ml.PfdModel
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.gms.vision.TfLiteVision
import org.tensorflow.lite.task.gms.vision.segmenter.ImageSegmenter
import org.tensorflow.lite.task.gms.vision.segmenter.OutputType
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class PFDHelper(val context: Context) {
    private lateinit var pfdModel:PfdModel
    private var numThreads: Int = 2
    private var currentDelegate: Int = 0
    private var imageSegmenter: ImageSegmenter? = null

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
            }.addOnFailureListener{
//                objectDetectorListener.onError("TfLiteVision failed to initialize: "
//                        + it.message)
            }
        }
    }

    private fun yuvToRgb(image: Image): Bitmap {
        val width = image.width
        val height = image.height
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val yBuffer = image.planes[0].buffer // Y
        val uBuffer = image.planes[1].buffer // U
        val vBuffer = image.planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // Add Y values
        yBuffer.get(nv21, 0, ySize)

        // Add VU values
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val imageBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    fun imageProxyToTensorBuffer(image: Image, inputSize: Int): TensorBuffer {
        val rgbBitmap = yuvToRgb(image)
        val resizedBitmap = Bitmap.createScaledBitmap(rgbBitmap, inputSize, inputSize, false)
        val byteBuffer = ByteBuffer.allocateDirect(resizedBitmap.width * resizedBitmap.height * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(resizedBitmap.width * resizedBitmap.height)
        resizedBitmap.getPixels(pixels, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        var pixel = 0
        for (i in 0 until resizedBitmap.width) {
            for (j in 0 until resizedBitmap.height) {
                val pixelVal = pixels[pixel++]
                byteBuffer.putFloat(Color.red(pixelVal) / 255.0f)
                byteBuffer.putFloat(Color.green(pixelVal) / 255.0f)
                byteBuffer.putFloat(Color.blue(pixelVal) / 255.0f)
            }
        }
        var fixedTensorBuffer: TensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, inputSize, inputSize, 3), DataType.FLOAT32)
        fixedTensorBuffer.loadBuffer(byteBuffer)

        return fixedTensorBuffer
    }

    @ExperimentalGetImage
    fun pfdInference(context: Context, imageProxy: ImageProxy): PfdModel.Outputs {
        // declare tensorflow model
        pfdModel = PfdModel.newInstance(context)

        // target image size to feed to model
        val targetImgSize = 512

        // image processing
        val bitmap: Bitmap = imageProxyToBitmap(imageProxy)

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    targetImgSize,
                    targetImgSize,
                    ResizeOp.ResizeMethod.NEAREST_NEIGHBOR
                )
            ) //.add(new Rot90Op(numRotation))
            .build()

        val tensorImageInput = imageProcessor.process(tensorImage)

        // Creates inputs for reference
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 1, 3), DataType.FLOAT32)

        val inputFeature1 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 14), DataType.FLOAT32)
//        inputFeature1.loadBuffer(byteBuffer)
        val inputFeature2 = TensorBuffer.createFixedSize(intArrayOf(1, 1, 4), DataType.FLOAT32)
//        inputFeature2.loadBuffer(byteBuffer)

        // run through model
        val outputs = pfdModel.process(inputFeature0, inputFeature1, inputFeature2)

        return outputs
    }

    fun perspectiveTransformation() {

    }

    fun destroyModel() {
        // Releases model resources if no longer used.
        pfdModel.close()
    }
}