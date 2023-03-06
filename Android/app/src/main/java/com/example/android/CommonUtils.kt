package com.example.android

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Point
import android.graphics.Rect
import android.media.MediaMetadataRetriever
import android.media.ThumbnailUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.PixelCopy
import android.view.SurfaceView
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.abs

enum class ModelType(val modelName: String) {
    HAR("har_gcn"), PFD("keypoint_rcnn"), VAD("silero_vad")
}

class CommonUtils(val context: Context) {
    // PFD Utils
    fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    fun readModel(modelType: ModelType): ByteArray {
        val modelID = context.resources.getIdentifier(modelType.modelName, "raw", context.packageName)
        return context.resources.openRawResource(modelID).readBytes()
    }

    fun matToBitmap(mat: Mat): Bitmap? {
        // Create a new Bitmap object
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)

        // Convert the Mat object to a Bitmap object using OpenCV's Utils class
        Utils.matToBitmap(mat, bitmap)

        return bitmap
    }

    @Throws(IOException::class)
    fun getModelByteBuffer(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun getFrameBitmap(view: SurfaceView, callback: (Bitmap?) -> Unit) {
        val bitmap: Bitmap = Bitmap.createBitmap(
            view.width,
            view.height,
            Bitmap.Config.ARGB_8888
        );

        try {
            // Create a handler thread to offload the processing of the image.
            val handlerThread = HandlerThread("PixelCopier");
            handlerThread.start();
            PixelCopy.request(
                view, bitmap,
                PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                    if (copyResult == PixelCopy.SUCCESS) {
                        callback(bitmap)
                    }
                    handlerThread.quitSafely();
                },
                Handler(handlerThread.looper)
            )
        } catch (e: IllegalArgumentException) {
            callback(null)
            // PixelCopy may throw IllegalArgumentException, make sure to handle it
            e.printStackTrace()
        }
    }

    fun resizeBitmap(bitmap: Bitmap, targetImgSize: Size): Bitmap {
        // resize bitmap
        return Bitmap.createScaledBitmap(bitmap, targetImgSize.width, targetImgSize.height, false)
    }

    data class BboxCenter(
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float
    )

    fun calculateBboxCenter(bbox: Array<FloatArray>): BboxCenter {
        val bboxWidth = abs(bbox[0][1] - bbox[1][1])
        val bboxHeight = abs(bbox[0][0] - bbox[3][0])
        val centerX = bbox[0][0] + bboxWidth / 2
        val centerY = bbox[0][1] + bboxHeight / 2
        return BboxCenter(centerX, centerY, bboxWidth, bboxHeight)
    }

    fun getVideoFrames(metadataRetriever: MediaMetadataRetriever): List<Bitmap> {
        val videoFrames = mutableListOf<Bitmap>()

        val duration =
            metadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong() ?: 0
        val numFrames = duration / 1000 * 30 // Assumes 30 frames per second
        for (i in 0 until numFrames) {
            val frameTime = i * 1000L / 30 // Assumes 30 frames per second
            val frame = metadataRetriever.getFrameAtTime(
                frameTime,
                MediaMetadataRetriever.OPTION_CLOSEST_SYNC
            )
            val thumbnail =
                ThumbnailUtils.extractThumbnail(frame, 640, 480) // Resize for performance
            videoFrames.add(thumbnail)
        }
        metadataRetriever.release()
        return videoFrames
    }
}