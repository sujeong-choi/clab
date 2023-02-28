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

    private lateinit var keyPoints:  KeypointType<Float>
    private var isDrawn: Boolean = false
    private var radius: Int = 0
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 15f
    }
    private val path = Path()

    init {
//        keyPoints.value = mutableListOf()
    }

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        if (isDrawn) {
            path.reset()

            // Move to the first vertex
            path.moveTo(keyPoints.value[0][0], keyPoints.value[0][1])

            for (i in 1 until keyPoints.value.size) {
                val keypoint = keyPoints.value[i]
                path.lineTo(keypoint[0], keypoint[1])
            }

            path.close()
            canvas?.drawPath(path, paint)

        }
    }

    fun drawOverlay(newKeyPoints: KeypointType<Float>, newRadius: Int = 20) {
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