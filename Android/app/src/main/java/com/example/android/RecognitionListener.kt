package com.example.android

interface RecognitionListener {
    fun onResult(hypothesis: Float?)

    fun onError(exception: Exception?)
}