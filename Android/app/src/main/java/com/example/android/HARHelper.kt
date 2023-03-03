package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.google.mediapipe.formats.proto.LandmarkProto
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.math.exp


// show the frames on the video
// save the video
class HARHelper(val context: Context) {
    private lateinit var poseDetector: PoseDetector
    private val classificationExecutor: Executor = Executors.newSingleThreadExecutor()

    init {
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

                 println(pose.allPoseLandmarks)

                // TODO: HarInference
            }
    }

    fun harInference(inputSkeleton: MultiArray<Float, DN>, harSession: OrtSession, vadSession: OrtSession, env: OrtEnvironment): String {
        val shape = longArrayOf(1, 3, 144, 25, 2)
        val capacity = shape.reduce{acc, s -> acc * s}.toInt()

        val inputNameIterator = harSession.inputNames!!.iterator()
        val inputName0: String = inputNameIterator.next()

        val bb: ByteBuffer = ByteBuffer.allocateDirect(capacity * 4)
        bb.order(ByteOrder.nativeOrder())
        val floatBuffer: FloatBuffer = bb.asFloatBuffer()
        floatBuffer.put(inputSkeleton.toFloatArray())
        floatBuffer.position(0)

        env.use {
            val tensor0 = OnnxTensor.createTensor(env, floatBuffer, shape)
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap[inputName0] = tensor0

            val output = harSession?.run(inputMap)

            val prediction: OnnxTensor = output?.toList()?.get(0)?.toPair()?.second as OnnxTensor

            val predictionArray: FloatArray = FloatArray(prediction.floatBuffer.remaining())
            prediction.floatBuffer.get(predictionArray)

            // get vad prediction
            vadSession

            return getLabel(predictionArray)
        }
    }

    private fun getLabel(output: FloatArray): String{
        return try {
            val outProb = softmax(output)
            val big3 = arrayOf(
                outProb.sliceArray(0 until 18).sum(),
                outProb[18],
                outProb[19]
            )

            var top1Index = 0
            var topValue = big3[0]

            for (i in 1 until big3.size) {
                if (big3[i] > topValue) {
                    top1Index = i
                    topValue = big3[i]
                }
            }

            when (top1Index) {
                0 -> ""
                1 -> "Painting:" + big3[top1Index].toString() + "%"
                else -> "Interview:" + big3[top1Index].toString() + "%"
            }
        } catch (e: Exception) {
            e.message?.let { Log.v("HAR", it) }
            ""
        }
    }

    private fun softmax(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        var sum = 0.0f

        // Compute exponential of each element and sum them up
        for (i in input.indices) {
            output[i] = exp(input[i].toDouble()).toFloat()
            sum += output[i]
        }

        // Normalize by dividing each element by the sum
        for (i in output.indices) {
            output[i] /= sum
        }

        return output
    }

    //save skeleton data in one frame and append to (144,25,3) ndarray
    fun saveSkeletonData(poseLandmarks: LandmarkProto.LandmarkList): D2Array<Float>{
        var oneFrameSkeleton = mk.d2array(25,3){0.0f}
        var landmark = poseLandmarks.landmarkList
        for(i: Int in 0..24) {
            if (landmark[i].visibility >= 0.75) {
                oneFrameSkeleton[i, 0] = landmark[i].x
                oneFrameSkeleton[i, 1] = landmark[i].y
                oneFrameSkeleton[i, 2] = landmark[i].z
            }
        }
        return oneFrameSkeleton
    }

    fun convertSkeletonData(framesSkeleton: D3Array<Float>): MultiArray<Float, DN> {
        //dummy humman skeleton data to align the input dimension of the model
        val dummyHumanSkeleton = mk.d3array(144,25,3) {0.0f}
        val humansSkeleton = mk.stack(framesSkeleton,dummyHumanSkeleton)

        //Transpose for processing in multiInput
        val transposeSkeleton = humansSkeleton.transpose(3,1,2,0)

        return transposeSkeleton.expandDims(axis = 0)
    }

    fun vadInference() {

    }

    fun destroyModel() {
        // Releases model resources if no longer used.
    }

}