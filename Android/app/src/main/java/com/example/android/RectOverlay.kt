package com.example.android

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.util.Size
import android.view.View

/**
 * Rectangle Overlay to display keypoints
 */
class RectOverlay constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    lateinit var frameSize: Size
    private lateinit var keyPoints: MutableList<FloatArray>
    private lateinit var bbox: FloatArray
    private var isDrawn: Boolean = false
    private var enableBoundingBox: Boolean = false
    private var radius: Int = 15
    private val rectPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 10f
    }
    private val paint = Paint().apply {
        color = Color.GREEN
    }
    private val rect = Rect()

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        if (isDrawn) {

            // resize keypoints to screensize
            val resizedKeypoints = resizeKeypoints(bbox, keyPoints)

            val localBbox = resizedKeypoints.bbox.value[0]
            val localKeypoint = resizedKeypoints.keypoint.value[0]

            for (keypoint in localKeypoint) {
                val kx = keypoint[0].toInt()
                val ky = keypoint[1].toInt()
                canvas?.drawCircle(kx.toFloat(), ky.toFloat(), radius.toFloat(), paint)
            }

            // for drawing a bounding box
            if (enableBoundingBox) {
                // get max and min x and y values
                val minX = localKeypoint.minByOrNull { it[0] }?.get(0) ?: 0.0f
                val minY = localKeypoint.minByOrNull { it[1] }?.get(1) ?: 0.0f
                val maxX = localKeypoint.maxByOrNull { it[0] }?.get(0) ?: 0.0f
                val maxY = localKeypoint.maxByOrNull { it[1] }?.get(1) ?: 0.0f

                canvas?.drawRect(
                    minX,
                    minY,
                    maxX,
                    maxY,
                    rectPaint
                )
            }
            canvas?.drawRect(rect, paint)
        }
    }

    fun drawKeypoints(
        newKeyPoints: MutableList<FloatArray>,
        newBbox: FloatArray,
        enableBbox: Boolean = false,
        newRadius: Int
    ) {
        enableBoundingBox = enableBbox
        keyPoints = newKeyPoints
        bbox = newBbox
        radius = newRadius
        isDrawn = true
        invalidate()
    }

    fun setScreenSize(size: Size) {
        frameSize = size
    }

    /**
     * Resizes the bounding box and keypoints according to the target Mediapipe resolution.
     * @param bbox The bounding box of the person in the frame as a FloatArray.
     * @param keypoints The list of key points detected on the person in the frame.
     * @return PfdResult object containing the resized bounding box and key points.
     */
    fun resizeKeypoints(
        bbox: FloatArray,
        keypoints: MutableList<FloatArray>
    ): PfdResult {
        val resizedOutput = PfdResult()

        // custom image size
        val widthRatio = frameSize.width / GlobalVars.targetMediapipeRes.width.toFloat()
        val heightRatio = frameSize.height / GlobalVars.targetMediapipeRes.height.toFloat()

        // resize bbox and keypoint
        resizedOutput.bbox.value.add(
            floatArrayOf(
                bbox[0] * widthRatio,
                bbox[1] * heightRatio,
                bbox[2] * widthRatio,
                bbox[3] * heightRatio
            )
        )

        val tempKp: MutableList<FloatArray> = mutableListOf()

        for (kp in keypoints) {
            tempKp.add(
                floatArrayOf(
                    kp[0] * widthRatio,
                    kp[1] * heightRatio
                )
            )
        }
        resizedOutput.keypoint.value.add(tempKp)

        return resizedOutput
    }

    fun clear() {
        rect.setEmpty()
        isDrawn = false
        invalidate()
    }
}