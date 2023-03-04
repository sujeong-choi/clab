package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.google.common.math.DoubleMath
import com.google.mediapipe.formats.proto.LandmarkProto
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
    private var isListening: Boolean = false
    private val classificationExecutor: Executor = Executors.newSingleThreadExecutor()
    private var previousLabel = 0
    private var previousVad = 0
    private var updatedLabel = 0
    private var updatedVad = 0
    private var vadProb: Float = 0f

    // vad variables
    private val vadService by lazy {
        VadService(context)
    }


    fun harInference(inputSkeleton: MultiArray<Float, DN>, harSession: OrtSession): String {
        val shape = longArrayOf(1, 3, 144, 25, 2)
        val capacity = shape.reduce { acc, s -> acc * s }.toInt()

        val inputNameIterator = harSession.inputNames!!.iterator()
        val inputName0: String = inputNameIterator.next()

        val bb: ByteBuffer = ByteBuffer.allocateDirect(capacity * 4)
        bb.order(ByteOrder.nativeOrder())
        val floatBuffer: FloatBuffer = bb.asFloatBuffer()
        floatBuffer.put(inputSkeleton.toFloatArray())
        floatBuffer.position(0)

        val env = GlobalVars.ortEnv

        env.use {
            val tensor0 = OnnxTensor.createTensor(env, floatBuffer, shape)
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap[inputName0] = tensor0

            val output = harSession.run(inputMap)

            val prediction: OnnxTensor = output?.toList()?.get(0)?.toPair()?.second as OnnxTensor

            val predictionArray: FloatArray = FloatArray(prediction.floatBuffer.remaining())
            prediction.floatBuffer.get(predictionArray)

            return getLabel(predictionArray, vadProb)
        }
    }

    private fun getLabel(harOutput: FloatArray, vadOutput: Float): String {
        return try {
            val harOutProb = softmax(harOutput)
            val big3 = arrayOf(
                harOutProb.sliceArray(0 until 17).max(),
                harOutProb[17],
                harOutProb[18]
            )

            var top1Index = 0
            var topValue = big3[0]

            for (i in 1 until big3.size) {
                if (big3[i] > topValue) {
                    top1Index = i
                    topValue = big3[i]
                }
            }

            val isVoiceDetected = if (vadOutput > 0.5f) 1 else 0

           if (updatedVad != isVoiceDetected && isVoiceDetected == previousVad) {
               updatedVad = isVoiceDetected
           }
           previousVad = isVoiceDetected

            if (top1Index == 2 && isVoiceDetected == 0) {
                top1Index = 0
            }

           if (top1Index != updatedLabel && top1Index == previousLabel) {
               updatedLabel = top1Index
           }
           previousLabel = top1Index

            when (updatedVad) {
                0 -> "Other"
                1 -> "Painting: " + String.format("%.1f", (big3[updatedVad] * 100)) + "%"
                else -> "Interview: " + String.format("%.1f", (big3[updatedVad] * 100)) + "%"
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
    fun saveSkeletonData(poseLandmarks: LandmarkProto.LandmarkList): D2Array<Float> {
        var oneFrameSkeleton = mk.d2array(25, 3) { 0.0f }
        var landmark = poseLandmarks.landmarkList
        for (i: Int in 0..24) {
            if (landmark[i].visibility >= 0.75) {
                oneFrameSkeleton[i, 0] = landmark[i].x
                oneFrameSkeleton[i, 1] = landmark[i].y
                oneFrameSkeleton[i, 2] = landmark[i].z
            }
        }
        return oneFrameSkeleton
    }

    private fun sampleSkeletonData(skeletonBuffer: ArrayList<D2Array<Float>>):D3Array<Float>{
        val framesSkeleton = mk.d3array(144, 25, 3) { 0.0f }
        val frameNum = skeletonBuffer.size

        if(frameNum>=60) {
            val skipInterval = frameNum / 60.0
            for (i in 0 until 60)
                framesSkeleton[i] = skeletonBuffer[(i * skipInterval).toInt()]
        }
        else
            for(i in 0 until frameNum)
                framesSkeleton[i] = skeletonBuffer[i]
        return framesSkeleton
    }

    fun convertSkeletonData(skeletonBuffer: ArrayList<D2Array<Float>>): MultiArray<Float, DN> {

        val framesSkeleton = sampleSkeletonData(skeletonBuffer)
        //dummy humman skeleton data to align the input dimension of the model
        val dummyHumanSkeleton = mk.d3array(144, 25, 3) { 0.0f }
        val humansSkeleton = mk.stack(framesSkeleton, dummyHumanSkeleton)

        //Transpose for processing in multiInput
        val transposeSkeleton = humansSkeleton.transpose(3, 1, 2, 0)

        return transposeSkeleton.expandDims(axis = 0)
    }

    private val recognitionListener = object : RecognitionListener {
        override fun onResult(hypothesis: Float?) {
            if (hypothesis != null)
                vadProb = hypothesis
        }

        override fun onError(exception: Exception?) {
            Log.e("VoiceTrigger", "Error", exception)
        }
    }

    fun vadInference() {
        if (!isListening) {
            vadService.startListening(recognitionListener)
        } else {
            vadService.stop()
        }
        isListening = !isListening
    }

    fun destroyModel() {
        // Releases model resources if no longer used.
    }

}