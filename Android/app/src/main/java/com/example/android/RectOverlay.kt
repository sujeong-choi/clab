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

    private lateinit var frameSize: Size
    private val targetSize: Int = 512
    private lateinit var keyPoints: MutableList<FloatArray>
    private lateinit var bbox: FloatArray
    private var isDrawn: Boolean = false
    private var enableBoundingBox: Boolean = false
    private var radius: Int = 25
    private val rectPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 10f
    }
    private val paint = Paint().apply {
        color = Color.WHITE
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
                canvas?.drawRect(
                    localBbox[0],
                    localBbox[1],
                    localBbox[2],
                    localBbox[3],
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

    fun resizeKeypoints(
        bbox: FloatArray,
        keypoints: MutableList<FloatArray>
    ): PfdResult {
        val resizedOutput = PfdResult()

        // custom image size
        val widthRatio = frameSize.width / targetSize.toFloat()
        val heightRatio = frameSize.height / targetSize.toFloat()

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