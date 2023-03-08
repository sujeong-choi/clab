package com.example.android

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

/**
 * Rectangle Overlay to display keypoints
 */
class RectOverlay constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    private lateinit var keyPoints: MutableList<FloatArray>
    private lateinit var bbox: FloatArray
    private var isDrawn: Boolean = false
    private var enableBoundBox: Boolean = false
    private var radius: Int = 15
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

            for (keypoint in keyPoints) {
                val kx = keypoint[0].toInt()
                val ky = keypoint[1].toInt()
                canvas?.drawCircle(kx.toFloat(), ky.toFloat(), radius.toFloat(), paint)
            }

            // for drawing a bounding box
            if (enableBoundBox) {
                canvas?.drawRect(
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
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
        newRadius: Int = 15
    ) {
        enableBoundBox = enableBbox
        keyPoints = newKeyPoints
        bbox = newBbox
        radius = newRadius
        isDrawn = true
        invalidate()
    }

    fun clear() {
        rect.setEmpty()
        isDrawn = false
        invalidate()
    }
}