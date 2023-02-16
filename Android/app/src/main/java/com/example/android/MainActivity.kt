package com.example.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.media.Image
import android.os.Bundle
import android.util.Size as UtilSize
import androidx.annotation.NonNull
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import com.example.android.databinding.ActivityMainBinding
import com.google.common.util.concurrent.ListenableFuture
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), CameraXConfig.Provider {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var videoView: PreviewView

    lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private val REQUEST_CODE_PERMISSION = 101
    private val REQUIRED_PERMISSIONS: Array<String> = arrayOf("android.permission.CAMERA")
    private val executor: Executor = Executors.newSingleThreadExecutor()

    override fun getCameraXConfig(): CameraXConfig {
        return Camera2Config.defaultConfig()
    }

    @ExperimentalGetImage
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        OpenCVLoader.initDebug()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        videoView = findViewById(R.id.videoView)

        setSupportActionBar(binding.toolbar)

        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)
        setupActionBarWithNavController(navController, appBarConfiguration)

        if(!checkPermissions()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION)
        }

        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable{
            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()
            startCamera(cameraProvider)

        }, ContextCompat.getMainExecutor(this))

    }

    private fun checkPermissions():Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ActivityCompat.checkSelfPermission(this, permission)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }

    // Create an OrtSession with the given OrtEnvironment
    private fun createORTSession(ortEnvironment: OrtEnvironment ) : OrtSession {
        val inputStream: InputStream = resources.openRawResource(R.raw.keypoint_rcnn_quant)
        val byteArrayOutputStream = ByteArrayOutputStream()
        val buffer = ByteArray(1024)
        var len = inputStream.read(buffer)
        while (len != -1) {
            byteArrayOutputStream.write(buffer, 0, len)
            len = inputStream.read(buffer)
        }
        val modelBytes = byteArrayOutputStream.toByteArray()
        inputStream.close()
        return ortEnvironment.createSession( modelBytes )
    }

    // Make predictions with given inputs
    private fun runPrediction( image: Image , ortSession: OrtSession , ortEnvironment: OrtEnvironment ) : Float {
        // Get the name of the input node
        val inputName = ortSession.inputNames?.iterator()?.next()

        // convert image to an OnnxTensor
        val inputTensor = OnnxTensor.createTensor( ortEnvironment , resizeImage(image) , longArrayOf( 512, 512 ) )

        // Run the model
        val results = ortSession.run( mapOf( inputName to inputTensor ) )

        // Fetch and return the results
        val output = results[0].value as Array<FloatArray>
        println("Model output")
        println(output)
//        return output[0][0]
        return 0.1F1
    }

    private fun resizeImage(image: Image, size: Size = Size(512.0, 512.0)): FloatBuffer {
        // Make a FloatBuffer of the inputs
        // Convert the Bitmap object to a Mat object
        val mat = Mat(image.height, image.width, CvType.CV_8UC4)

        val bitmap = Bitmap.createBitmap(image.height, image.width, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint()
        canvas.drawBitmap(bitmap, 0f, 0f, paint)
//
        Utils.bitmapToMat(bitmap, mat)

        val resizedImage = Mat()
        val size = Size(512.0, 512.0)
        Imgproc.resize(mat, resizedImage, size)

//        val buffer: ByteBuffer = image.planes[0].buffer
        val buffer: ByteBuffer = ByteBuffer.allocate((resizedImage.total()).toInt())
        resizedImage.get(0, 0, buffer.array())
        println(buffer.asFloatBuffer())

        return buffer.asFloatBuffer()
    }



    @ExperimentalGetImage
    private fun startCamera(@NonNull cameraProvider: ProcessCameraProvider ) {
        // Set up the preview use case to display camera preview.
        val preview = Preview.Builder().build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        preview.setSurfaceProvider(videoView.surfaceProvider)

        val imageAnalysis: ImageAnalysis = ImageAnalysis.Builder().setTargetResolution(UtilSize(1280, 720)).build()

        val ortEnvironment = OrtEnvironment.getEnvironment()
        val ortSession = createORTSession(ortEnvironment)

        imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            // insert your code here.
            // after done, release the ImageProxy object
            val output = runPrediction( imageProxy.image!!, ortSession , ortEnvironment )

            imageProxy.close()
        })

        val camera: Camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
    }
}