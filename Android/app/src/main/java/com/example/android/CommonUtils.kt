package com.example.android

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.media.ThumbnailUtils
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.PixelCopy
import android.view.SurfaceView
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.abs

class GlobalVars {
    companion object {
        @JvmField
        // setup global onnx model environment
        var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
        var targetMediapipeRes: Size = Size(0, 0)
    }
}

// all model naming is done here
enum class ModelType(val modelName: String) {
    HAR("har_gcn"), PFD("keypoint_rcnn"), VAD("silero_vad")
}

/**
 * Common Utils that are used throughout the app
 */
class CommonUtils(val context: Context) {
    /**
     * Converts a bitmap image to a OpenCV Mat object.
     * @param bitmap The bitmap image to be converted.
     * @return The corresponding OpenCV Mat object.
     */
    fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bitmap, mat)
        return mat
    }

    /**
     * Reads a model file and returns it as a byte array. The model file is identified by its name which is
     * specified by the given [modelType] parameter. The method looks for the model file in the 'raw' resource
     * folder of the app and reads its contents into a byte array.
     * @param modelType a [ModelType] enum representing the model file to read.
     * @return a byte array containing the contents of the model file.
     */
    fun readModel(modelType: ModelType): ByteArray {
        val modelID = context.resources.getIdentifier(modelType.modelName, "raw", context.packageName)
        return context.resources.openRawResource(modelID).readBytes()
    }

    /**
     * Convert a OpenCV Mat object to a Bitmap object.
     * @param mat The Mat object to convert to a Bitmap object
     * @return The converted Bitmap object
     */
    fun matToBitmap(mat: Mat): Bitmap? {
        // Create a new Bitmap object
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)

        // Convert the Mat object to a Bitmap object using OpenCV's Utils class
        Utils.matToBitmap(mat, bitmap)

        return bitmap
    }

    /**
     * Captures a bitmap of the current frame displayed on a SurfaceView and passes it to a callback function
     * along with a timestamp of when the bitmap was captured.
     * @param view The SurfaceView to capture the frame from.
     * @param callback A function to be called once the bitmap is captured, which takes in the bitmap and the timestamp.
     */
    fun getFrameBitmap(view: SurfaceView, callback: (Bitmap?, Long) -> Unit) {
        val bitmap: Bitmap = Bitmap.createBitmap(
            view.width,
            view.height,
            Bitmap.Config.ARGB_8888
        )

        try {
            // Create a handler thread to offload the processing of the image.
            val handlerThread = HandlerThread("PixelCopier")
            handlerThread.start()
            val bitmapTimestamp = System.currentTimeMillis()
            PixelCopy.request(
                view, bitmap,
                PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                    if (copyResult == PixelCopy.SUCCESS) {
                        callback(bitmap, bitmapTimestamp)
                    }
                    handlerThread.quitSafely()
                },
                Handler(handlerThread.looper)
            )
        } catch (e: IllegalArgumentException) {
            callback(null, System.currentTimeMillis())
            // PixelCopy may throw IllegalArgumentException, make sure to handle it
            e.printStackTrace()
        }
    }

    /**
     * Resizes the input bitmap to a square image with a target size.
     * @param bitmap The input bitmap to be resized.
     * @param targetImgSize The target size for the output bitmap.
     * @return The resized square bitmap.
     */
    fun resizeBitmap(bitmap: Bitmap, targetImgSize: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, targetImgSize, targetImgSize, false)
    }

    data class BboxCenter(
        val centerX: Float,
        val centerY: Float,
        val width: Float,
        val height: Float
    )

    /**
     * Calculates the center of a bounding box represented by an array of four coordinate points (top-left, top-right, bottom-right, bottom-left)
     * @param bbox Array of four coordinate points that define the bounding box. Each coordinate point is represented as a float array [y, x].
     * @return A [BboxCenter] object that contains the center coordinates (x, y), width, and height of the bounding box.
     */
    fun calculateBboxCenter(bbox: Array<FloatArray>): BboxCenter {
        val bboxWidth = abs(bbox[0][1] - bbox[1][1])
        val bboxHeight = abs(bbox[0][0] - bbox[3][0])
        val centerX = bbox[0][0] + bboxWidth / 2
        val centerY = bbox[0][1] + bboxHeight / 2
        return BboxCenter(centerX, centerY, bboxWidth, bboxHeight)
    }

    /**
     * Extracts frames from a video using MediaMetadataRetriever and returns them as a list of bitmaps.
     * @param metadataRetriever An instance of MediaMetadataRetriever to extract frames from.
     * @return A list of bitmaps representing frames from the video.
     */
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