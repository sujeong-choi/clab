package com.example.android

import android.content.Context
import android.graphics.*
import android.media.Image
import android.media.MediaMetadataRetriever
import android.media.ThumbnailUtils
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.common.internal.ImageConvertUtils
import com.google.protobuf.InvalidProtocolBufferException
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.ByteArrayOutputStream
import kotlin.math.abs


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
    fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    fun matToBitmap(mat: Mat): Bitmap? {
        // Create a new Bitmap object
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)

        // Convert the Mat object to a Bitmap object using OpenCV's Utils class
        Utils.matToBitmap(mat, bitmap)

        return bitmap
    }

    @ExperimentalGetImage
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val image: InputImage =
            InputImage.fromMediaImage(imageProxy.image!!, imageProxy.imageInfo.rotationDegrees);

        return ImageConvertUtils.getInstance().getUpRightBitmap(image)
            .copy(Bitmap.Config.ARGB_8888, true)
    }

    data class BboxCenter(val centerX: Float, val centerY: Float, val width: Float, val height: Float)

    fun calculateBboxCenter(bbox: Array<FloatArray>): BboxCenter {
        val bboxWidth = abs(bbox[0][1] - bbox[1][1])
        val bboxHeight = abs(bbox[0][0] - bbox[3][0])
        val centerX = bbox[0][0] + bboxWidth / 2
        val centerY = bbox[0][1] + bboxHeight / 2
        return BboxCenter(centerX, centerY, bboxWidth, bboxHeight)
    }

    fun getVideoFrames(metadataRetriever: MediaMetadataRetriever): List<Bitmap> {
        val videoFrames = mutableListOf<Bitmap>()

        val duration = metadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0
        val numFrames = duration / 1000 * 30 // Assumes 30 frames per second
        for (i in 0 until numFrames) {
            val frameTime = i * 1000L / 30 // Assumes 30 frames per second
            val frame = metadataRetriever.getFrameAtTime(frameTime, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
            val thumbnail = ThumbnailUtils.extractThumbnail(frame, 640, 480) // Resize for performance
            videoFrames.add(thumbnail)
        }
        metadataRetriever.release()
        return videoFrames
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