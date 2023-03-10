package com.example.android

import android.content.Context
import android.util.AttributeSet
import android.widget.MediaController

class CustomMediaController : MediaController {
    constructor(context: Context?, attrs: AttributeSet?) : super(context, attrs) {}
    constructor(context: Context?, useFastForward: Boolean) : super(
        context,
        useFastForward
    ) {
    }

    constructor(context: Context?) : super(context) {}

    override fun hide() {
        // don't hide
    }
}