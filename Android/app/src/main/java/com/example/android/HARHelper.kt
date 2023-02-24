package com.example.android

import android.content.Context
import com.example.android.ml.HarModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class HARHelper {
    private lateinit var poseDetector: PoseDetector
    private val classificationExecutor: Executor = Executors.newSingleThreadExecutor()
    private lateinit var harModel: HarModel

    constructor() {
        // Pose detector
        val options = PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
            .build()

        poseDetector = PoseDetection.getClient(options)
    }


    fun detectInImage(image: InputImage) {
        poseDetector
            .process(image)
            .continueWith(
                classificationExecutor
            ) { task ->
                val pose = task.result

                // TODO: HAR Model Implementation
                // println(pose.allPoseLandmarks)
            }
    }

    fun harInference(context: Context, inputImage: InputImage) {
        try {
            harModel = HarModel.newInstance(context)
        }
        catch (e: Exception) {
            println(e.message)
        }
        // byte buffer
        val byteBuffer = inputImage.byteBuffer!!

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(16, 3, 6, 144, 25, 2), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = harModel.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val outputFeature1 = outputs.outputFeature1AsTensorBuffer
    }

    fun vadInference() {

    }

    fun destroyModel() {
        // Releases model resources if no longer used.
        harModel.close()
    }

}