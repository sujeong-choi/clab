package com.example.android
import android.graphics.SurfaceTexture

class CustomSurfaceTexture(texName: Int) : SurfaceTexture(texName) {
    private fun init() {
        super.detachFromGLContext()
    }

    init {
        init()
    }
}