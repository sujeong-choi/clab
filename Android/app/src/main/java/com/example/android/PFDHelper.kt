package com.example.android

import android.content.Context
import android.content.res.AssetManager
import android.graphics.*
import android.media.Image
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.example.android.ml.KeypointModel
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
    private lateinit var keypointModel: KeypointModel
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
            }.addOnFailureListener {
//                objectDetectorListener.onError("TfLiteVision failed to initialize: "
//                        + it.message)
            }
        }
    }


    @ExperimentalGetImage
    fun pfdInference(context: Context, imageProxy: ImageProxy): TensorBuffer {
        // declare tensorflow model
        keypointModel = KeypointModel.newInstance(context)

        // target image size to feed to model
        val targetImgSize = 512

        // image processing
        val bitmap: Bitmap = LocalUtils(context).imageProxyToBitmap(imageProxy)

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

        return outputBuffer
    }

    fun perspectiveTransformation(imageProxy: ImageProxy, keyPoints: Any) {

    }

    fun destroyModel() {
        // Releases model resources if no longer used.
        keypointModel.close()
    }
}