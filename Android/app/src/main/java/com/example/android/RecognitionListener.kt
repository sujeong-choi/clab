package com.example.android

interface RecognitionListener {
    fun onResult(hypothesis: String?)

    fun onError(exception: Exception?)
}