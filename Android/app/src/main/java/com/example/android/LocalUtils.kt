package com.example.android

import android.content.Context
import android.graphics.*
import android.media.Image
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import com.google.protobuf.InvalidProtocolBufferException
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder


class LocalUtils(context: Context) {
    private val TAG = "YXH"
    private val BINARY_GRAPH_NAME = "pose_tracking_gpu.binarypb"
    private val INPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_VIDEO_STREAM_NAME = "output_video"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "pose_landmarks"
    private val FLIP_FRAMES_VERTICALLY = true
    private var eglManager: EglManager
    var frameProcessor: FrameProcessor
    private lateinit var converter: ExternalTextureConverter

    init {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni")
        System.loadLibrary("opencv_java3")


        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(context)
        eglManager = EglManager(null)
        frameProcessor = FrameProcessor(
            context,
            eglManager.nativeContext,
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME
        )
        frameProcessor
            .videoSurfaceOutput
            .setFlipY(FLIP_FRAMES_VERTICALLY)

        //        inputSidePackets.put(INPUT_NUM_HANDS_SIDE_PACKET_NAME, packetCreator.createInt32(NUM_HANDS));
//        processor.setInputSidePackets(inputSidePackets);

        // To show verbose logging, run:
        // adb shell setprop log.tag.MainActivity VERBOSE
//        if (Log.isLoggable(TAG, Log.VERBOSE)) {
        frameProcessor.addPacketCallback(
            OUTPUT_LANDMARKS_STREAM_NAME
        ) { packet: Packet ->
            Log.v(TAG, "Received multi-hand landmarks packet.")
            Log.v(TAG, packet.toString())
            val landmarksRaw = PacketGetter.getProtoBytes(packet)
            try {
                val landmarks = NormalizedLandmarkList.parseFrom(landmarksRaw)
                if (landmarks == null) {
                    Log.v(
                        TAG,
                        "[TS:" + packet.timestamp + "] No iris landmarks."
                    )
                    return@addPacketCallback
                }
                // Note: If eye_presence is false, these landmarks are useless.
                Log.v(
                    TAG,
                    "[TS:"
                            + packet.timestamp
                            + "] #Landmarks for iris: "
                            + landmarks.landmarkCount
                )
                getLandmarksDebugString(landmarks)?.let { Log.v(TAG, it) }
            } catch (e: InvalidProtocolBufferException) {
                Log.e(TAG, "Couldn't Exception received - $e")
                return@addPacketCallback
            }
        }
    }

    // needs to be invoked before camera is started
    fun initExternalTextureConverter(): ExternalTextureConverter {
        converter = ExternalTextureConverter(
            eglManager.context, 2
        )
        converter.setFlipY(FLIP_FRAMES_VERTICALLY)
        converter.setConsumer(frameProcessor)
        return converter
    }

    // PFD Utils
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

    fun imageProxyToTensorBuffer(image: Image, inputSize: Int): TensorBuffer {
        val rgbBitmap = yuvToRgb(image)
        val resizedBitmap = Bitmap.createScaledBitmap(rgbBitmap, inputSize, inputSize, false)
        val byteBuffer = ByteBuffer.allocateDirect(resizedBitmap.width * resizedBitmap.height * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(resizedBitmap.width * resizedBitmap.height)
        resizedBitmap.getPixels(
            pixels,
            0,
            resizedBitmap.width,
            0,
            0,
            resizedBitmap.width,
            resizedBitmap.height
        )

        var pixel = 0
        for (i in 0 until resizedBitmap.width) {
            for (j in 0 until resizedBitmap.height) {
                val pixelVal = pixels[pixel++]
                byteBuffer.putFloat(Color.red(pixelVal) / 255.0f)
                byteBuffer.putFloat(Color.green(pixelVal) / 255.0f)
                byteBuffer.putFloat(Color.blue(pixelVal) / 255.0f)
            }
        }
        var fixedTensorBuffer: TensorBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, inputSize, inputSize, 3), DataType.FLOAT32)
        fixedTensorBuffer.loadBuffer(byteBuffer)

        return fixedTensorBuffer
    }

    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
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

    // HAR Utils
    private fun getLandmarksDebugString(landmarks: NormalizedLandmarkList): String? {
        var landmarkIndex = 0
        var landmarksString = ""
        for (landmark in landmarks.landmarkList) {
            landmarksString += "Landmark[$landmarkIndex]: (${landmark.x}, ${landmark.y}, ${landmark.z})"
            ++landmarkIndex
        }
        return landmarksString
    }
}