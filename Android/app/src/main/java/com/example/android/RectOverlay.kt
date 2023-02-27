package com.example.android

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.View

class RectOverlay constructor(context: Context?, attributeSet: AttributeSet?) :
    View(context, attributeSet) {

    private var keyPoints: Array<FloatArray>
    private var isDrawn: Boolean = false
    private var radius: Int = 0
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 15f
    }
    private val path = Path()

    init {
        keyPoints = arrayOf(floatArrayOf())
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        if (isDrawn) {
            path.reset()

            // Move to the first vertex
            path.moveTo(keyPoints[0][0], keyPoints[0][1])

            for (i in 1 until keyPoints.size) {
                val keypoint = keyPoints[i]
                path.lineTo(keypoint[0], keypoint[1])
            }

            path.close()
            canvas?.drawPath(path, paint)

        }
    }

    fun drawOverlay(newKeyPoints: Array<FloatArray>, newRadius: Int = 20) {
        keyPoints = newKeyPoints
        radius = newRadius
        isDrawn = true
        invalidate()
    }

    fun clear() {
        path.reset()
        isDrawn = false
        invalidate()
    }
}