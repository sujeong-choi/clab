package com.example.android

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.time.ExperimentalTime

class VadService(
    val context: Context,
    sampleRate: Int = 16000,
    private val bufferSize: Int = 4000,
) {

    @SuppressLint("MissingPermission")
    private val recorder: AudioRecord = AudioRecord(
        MediaRecorder.AudioSource.VOICE_RECOGNITION, sampleRate, AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT, bufferSize * 2
    )

    private val recognizer: Recognizer = Recognizer()
    private var recognizerThread: RecognizerThread? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    /**
     * Starts the voice recognition service and sets the recognition listener to receive recognition results.
     * @param listener the listener to receive recognition results.
     * @return true if the service has been started successfully, false otherwise.
     */
    fun startListening(listener: RecognitionListener): Boolean {
        if (recognizerThread != null) return false
        recognizerThread = RecognizerThread(listener)
        recognizerThread!!.start()
        Log.d("SpeechService", "Recognition has been started")
        return true
    }

    /**
     * Stops the current recognizer thread if there is one running.
     * @return true if a recognizer thread was running and was successfully stopped, false otherwise.
     */
    private fun stopRecognizerThread(): Boolean {
        if (recognizerThread == null) return false
        try {
            recognizerThread!!.interrupt()
            recognizerThread!!.join()
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        recognizerThread = null
        Log.d("SpeechService", "Recognition has been stopped")
        return true
    }

    fun stop(): Boolean {
        return stopRecognizerThread()
    }

    fun cancel(): Boolean {
        recognizerThread?.setPause(true)
        return stopRecognizerThread()
    }

    fun shutdown() {
        recorder.stop()
        recorder.release()
    }

    fun setPause(paused: Boolean) {
        recognizerThread?.setPause(paused)
    }

    /**
     * Private inner class for running the speech recognition on a separate thread.
     * @param listener The RecognitionListener to send recognition results to.
     */
    private inner class RecognizerThread(
        var listener: RecognitionListener
    ) : Thread() {
        @Volatile
        private var paused = false

        fun setPause(paused: Boolean) {
            this.paused = paused
        }

        @ExperimentalTime
        override fun run() {
            recorder.startRecording()
            if (recorder.recordingState == AudioRecord.RECORDSTATE_STOPPED) {
                recorder.stop()
                val ioe = IOException(
                    "Failed to start recording. Microphone might be already in use."
                )
                mainHandler.post { listener.onError(ioe) }
            }
            val buffer = ShortArray(bufferSize)
            while (!interrupted()) {
                val nread = recorder.read(buffer, 0, buffer.size)
                if (paused) continue
                if (nread < 0) throw RuntimeException("Error reading audio buffer")
                val floatArray =
                    buffer.map { it.toFloat() / Short.MAX_VALUE.toFloat() }.toFloatArray()
                val numRes = recognizer.checkIfChunkHasVoice(floatArray)
                mainHandler.post { listener.onResult(numRes) }
            }
            recorder.stop()
        }
    }

    /**
     * The Recognizer class represents a PyTorch model that performs voice activity detection (VAD).
     * The class loads a pre-trained model from the assets folder and uses it to perform real-time
     * voice detection on incoming audio data. The class relies on the PyTorch Android API to load
     * and use the model.
     * @property context The context of the Android application that uses this class.
     * @property vadModule The PyTorch module that performs voice activity detection.
     * @constructor Creates an instance of the Recognizer class.
     */
    private inner class Recognizer() {
        private val vadModule: Module by lazy {
            loadModule("vad.jit").also {
                Log.d("PyTorch", "Vad module has been initialized")
            }
        }

        /**
         * Given an input float array buffer, uses a preloaded PyTorch module for Voice Activity Detection (VAD) to determine if the chunk has voice or not.
         * @param floatInputBuffer Input float array buffer to check for voice.
         * @return A float representing the probability that the chunk has voice.
         */
        fun checkIfChunkHasVoice(floatInputBuffer: FloatArray): Float {
            val result = vadModule.getResult(floatInputBuffer)
            return result.toTensor().dataAsFloatArray[1]
        }

        /**
         * Given a path to a PyTorch module file, loads the module and returns it.
         * @param path The path to the PyTorch module file.
         * @return The loaded PyTorch module.
         */
        private fun loadModule(path: String): Module {
            val modulePath = assetFilePath(context, path)
            val moduleFileAbsoluteFilePath = File(modulePath).absolutePath
            return Module.load(moduleFileAbsoluteFilePath)
        }

        /**
         * Returns the absolute file path of the given asset in the app's file directory.
         * If the file already exists, its path is returned. Otherwise, the asset is opened
         * and copied to the file directory before its path is returned.
         * @param context the context of the app
         * @param assetName the name of the asset file
         * @return the absolute file path of the asset in the app's file directory
         * @throws IOException if an I/O error occurs while reading the asset or writing to the file directory
         */
        private fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }

        /**
         * Gets the result from the module by passing in a float input buffer as a parameter.
         * The float input buffer is first converted to a Tensor object, which is then passed
         * to the forward method along with an IValue object. The resulting IValue object
         * is returned.
         * @param floatInputBuffer a FloatArray containing the input buffer.
         * @return the IValue object obtained from the forward method.
         */
        private fun Module.getResult(floatInputBuffer: FloatArray): IValue {
            val inTensorBuffer = Tensor.allocateFloatBuffer(floatInputBuffer.size)
            inTensorBuffer.put(floatInputBuffer)
            val inTensor =
                Tensor.fromBlob(inTensorBuffer, longArrayOf(1, floatInputBuffer.size.toLong()))
            return forward(IValue.from(inTensor))
        }
    }
}