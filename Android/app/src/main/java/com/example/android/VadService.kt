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

    // start recognizer thread ot listen to audio and make vad predictions
    fun startListening(listener: RecognitionListener): Boolean {
        if (recognizerThread != null) return false
        recognizerThread = RecognizerThread(listener)
        recognizerThread!!.start()
        Log.d("SpeechService", "Recognition has been started")
        return true
    }

    // stop recognizer thread to freeup resources
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

    private inner class Recognizer() {
        private val VAD_TRESHOLD = 0.5f

        private val vadModule: Module by lazy {
            loadModule("vad.jit").also {
                Log.d("PyTorch", "Vad module has been initialized")
            }
        }

        fun checkIfChunkHasVoice(floatInputBuffer: FloatArray): Float {
            val result = vadModule.getResult(floatInputBuffer)
            return result.toTensor().dataAsFloatArray[1]
        }

        private fun loadModule(path: String): Module {
            val modulePath = assetFilePath(context, path)
            val moduleFileAbsoluteFilePath = File(modulePath).absolutePath
            return Module.load(moduleFileAbsoluteFilePath)
        }

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

        private fun Module.getResult(floatInputBuffer: FloatArray): IValue {
            val inTensorBuffer = Tensor.allocateFloatBuffer(floatInputBuffer.size)
            inTensorBuffer.put(floatInputBuffer)
            val inTensor =
                Tensor.fromBlob(inTensorBuffer, longArrayOf(1, floatInputBuffer.size.toLong()))
            return forward(IValue.from(inTensor))
        }
    }
}