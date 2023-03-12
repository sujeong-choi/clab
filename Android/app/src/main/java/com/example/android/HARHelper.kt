package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.google.mediapipe.formats.proto.LandmarkProto
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.expandDims
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.jetbrains.kotlinx.multik.ndarray.operations.toFloatArray
import java.nio.FloatBuffer
import kotlin.math.exp


/**
 * HAR Helper class containing all logic related to HAR Inference and Skeleton Processing
 */
class HARHelper(val context: Context) {
    private var isListening: Boolean = false
    private var previousLabel = 0
    private var updatedLabel = 0
    private var vadProb: Float = 0f

    // vad variables
    private val vadService by lazy {
        VadService(context)
    }

    /**
     * Runs inference on a given input skeleton using a given HAR session and returns the predicted action label.
     * @param inputSkeleton The input skeleton represented as a MultiArray<Float, DN> object.
     * @param harSession The HAR session to use for inference.
     * @return The predicted action label as a string.
     */
    fun harInference(inputSkeleton: MultiArray<Float, DN>, harSession: OrtSession?): String {
        val shape = longArrayOf(1, 3, 144, 25, 2)

        val inputNameIterator = harSession?.inputNames!!.iterator()
        val inputName0: String = inputNameIterator.next()


        val floatArrayData = inputSkeleton.toFloatArray()
        val buffer = FloatBuffer.allocate(floatArrayData.size)
        buffer.put(floatArrayData)
        buffer.flip()

        val env = GlobalVars.ortEnv

        env.use {
            val tensor0 = OnnxTensor.createTensor(env, buffer, shape)
            val inputMap = mutableMapOf<String, OnnxTensor>()
            inputMap[inputName0] = tensor0

            val output = harSession.run(inputMap)

            val prediction: OnnxTensor = output?.toList()?.get(0)?.toPair()?.second as OnnxTensor

            val predictionArray: FloatArray = FloatArray(prediction.floatBuffer.remaining())
            prediction.floatBuffer.get(predictionArray)

            return getLabel(predictionArray, vadProb)
        }
    }

    /**
     * This function takes the output of a HAR (Human Activity Recognition) model and a VAD (Voice Activity Detection) model,
     * and returns a label string representing the recognized activity. It uses the softmax function to convert the
     * output of the HAR model into a probability distribution, then identifies the top three values and returns the label
     * associated with the highest value. If the highest value corresponds to an interview activity and there is no voice detected,
     * the function returns the "unknown" label. It also keeps track of the previous and updated label to smooth the output.
     * @param harOutput the output of the HAR model as a FloatArray
     * @param vadOutput the output of the VAD model as a Float value
     * @return a String representing the recognized activity
     */
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

            if (top1Index == 2 && isVoiceDetected == 0) {
                top1Index = 0
            }

            if (updatedLabel != top1Index && previousLabel == top1Index)
                updatedLabel = top1Index
            previousLabel = top1Index

            top1Index = updatedLabel

            when (top1Index) {
                0 -> " "
                1 -> "Painting: " + String.format("%.1f", (big3[top1Index] * 100)) + "%"
                else -> "Interview: " + String.format("%.1f", (big3[top1Index] * 100)) + "%"
            }
        } catch (e: Exception) {
            e.message?.let { Log.v("HAR", it) }
            ""
        }
    }

    /**
     * Computes the softmax activation function for the given input array.
     * @param input the input array of floats
     * @return the output array after applying softmax activation
     */
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

    /**
     * Saves the skeleton data from a single frame of pose landmarks.
     * @param poseLandmarks a LandmarkProto.LandmarkList containing pose landmarks for a single frame
     * @return a D2Array<Float> containing the skeleton data
     */
    fun saveSkeletonData(poseLandmarks: LandmarkProto.LandmarkList): D2Array<Float> {
        val oneFrameSkeleton = mk.d2array(25, 3) { 0.0f }
        val landmark = poseLandmarks.landmarkList
        for (i: Int in 0..24) {
            if (landmark[i].visibility >= 0.75) {
                oneFrameSkeleton[i, 0] = landmark[i].x
                oneFrameSkeleton[i, 1] = landmark[i].y
                oneFrameSkeleton[i, 2] = landmark[i].z
            }
        }
        return oneFrameSkeleton
    }

    /**
     * Samples skeleton data from skeletonBuffer to produce a 3D array of size (144, 25, 3).
     * The input skeletonBuffer contains an array list of individual skeleton frames, where each frame
     * contains 25 3D-coordinates (x, y, z) representing the pose landmarks.
     * This function downsamples the skeletonBuffer by taking only 60 frames, evenly distributed
     * over the length of the buffer. If the buffer has less than 60 frames, all frames are used.
     * @param skeletonBuffer An array list of individual skeleton frames containing pose landmarks.
     * @return A 3D array of size (144, 25, 3) representing the sampled skeleton data.
     */
    private fun sampleSkeletonData(skeletonBuffer: ArrayList<D2Array<Float>>): D3Array<Float> {
        val framesSkeleton = mk.d3array(144, 25, 3) { 0.0f }
        val frameNum = skeletonBuffer.size

        if (frameNum >= 60) {
            val skipInterval = frameNum / 60.0
            for (i in 0 until 60)
                framesSkeleton[i] = skeletonBuffer[(i * skipInterval).toInt()]
        } else
            for (i in 0 until frameNum)
                framesSkeleton[i] = skeletonBuffer[i]
        return framesSkeleton
    }

    /**
     * Converts a list of 2D arrays representing human skeleton data into a 5D tensor for input into the activity recognition model.
     * The input list must have at least 60 frames of skeleton data. If it has more than 60 frames, this function will sample
     * 60 frames uniformly from the input list. If it has fewer than 60 frames, the function will duplicate the frames to create
     * 60 frames of data.
     * The resulting tensor will have dimensions (1, 3, 144, 25, 2), where the first dimension is the batch size (always 1), the
     * second dimension is the number of channels, the third dimension is the time step, the fourth dimension is the number of joints,
     * and the fifth dimension is the x, y, and z coordinates for each joint.
     * @param skeletonBuffer List of 2D arrays representing human skeleton data
     * @return A 5D tensor of shape (1, 3, 144, 25, 2) containing the skeleton data in a format suitable for input into the activity recognition model
     */
    fun convertSkeletonData(skeletonBuffer: ArrayList<D2Array<Float>>): MultiArray<Float, DN> {
        //dummy humman skeleton data to align the input dimension of the model
        val humansSkeleton =
            mk.stack(sampleSkeletonData(skeletonBuffer), mk.d3array(144, 25, 3) { 0.0f })
        return humansSkeleton.transpose(3, 1, 2, 0).expandDims(axis = 0)
    }

    /**
     * An implementation of RecognitionListener interface used for handling events related to voice activity detection (VAD).
     * This listener updates the vadProb variable with the probability of voice activity detected by the VAD model.
     * @property vadProb a Float variable that holds the probability of voice activity detected by the VAD model.
     */
    private val recognitionListener = object : RecognitionListener {
        override fun onResult(hypothesis: Float?) {
            if (hypothesis != null)
                vadProb = hypothesis
        }

        override fun onError(exception: Exception?) {
            Log.e("VoiceTrigger", "Error", exception)
        }
    }

    /**
     * Triggers the Voice Activity Detection (VAD) service to start or stop listening based on its current state.
     * If it is not currently listening, it starts listening and sets the [recognitionListener] to receive the result.
     * If it is currently listening, it stops the VAD service.
     */
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