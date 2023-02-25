package com.example.android

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.AttributeSet
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import android.widget.VideoView
import androidx.annotation.NonNull
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import com.example.android.databinding.ActivityMainBinding
import com.google.common.util.concurrent.ListenableFuture
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import android.util.Size as UtilSize

class MainActivity : AppCompatActivity(), CameraXConfig.Provider {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var previewView: PreviewView
    private lateinit var previewViewSmall: ImageView
    private lateinit var videoView: VideoView
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private val REQUEST_CODE_PERMISSION = 101
    private val REQUIRED_PERMISSIONS: Array<String> =
        arrayOf("android.permission.CAMERA", "android.permission.READ_EXTERNAL_STORAGE")
    private val executor: Executor = Executors.newSingleThreadExecutor()
    private lateinit var harHelper: HARHelper
    private lateinit var pfdHelper: PFDHelper


    override fun getCameraXConfig(): CameraXConfig {
        return Camera2Config.defaultConfig()
    }

    @ExperimentalGetImage
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "Failed to initialize OpenCV", Toast.LENGTH_LONG).show()
        }

        pfdHelper = PFDHelper(this)
        harHelper = HARHelper()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        previewView = findViewById(R.id.previewView)
        previewViewSmall = findViewById(R.id.previewViewSmall)

        //set bg color for previewViewSmall
        previewViewSmall.setBackgroundColor(ContextCompat.getColor(this, android.R.color.darker_gray))

        videoView = findViewById(R.id.videoView)

        setSupportActionBar(binding.toolbar)

        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)
        setupActionBarWithNavController(navController, appBarConfiguration)


        if (!checkPermissions()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSION)
        }

        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()
            startCamera(cameraProvider)

        }, ContextCompat.getMainExecutor(this))

    }

    private fun checkPermissions(): Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ActivityCompat.checkSelfPermission(this, permission)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }

    @ExperimentalGetImage
    private fun startCamera(@NonNull cameraProvider: ProcessCameraProvider) {
        // Set up the preview use case to display camera preview.
        val preview = Preview.Builder().build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val imageAnalysis: ImageAnalysis =
            ImageAnalysis.Builder().setTargetResolution(UtilSize(1280, 720)).build()

        var lastAnalyzedTimestamp = 0L
        val predictionInterval = 5000L

        imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
            val currentTimestamp = System.currentTimeMillis()

            if (currentTimestamp - lastAnalyzedTimestamp >= predictionInterval) {
                lastAnalyzedTimestamp = System.currentTimeMillis()

                try {
                    // run mediapipe keypoint detection
                    LocalUtils(this).frameProcessor

//                harHelper.detectInImage(
//                    InputImage.fromMediaImage(
//                        imageProxy.image!!,
//                        imageProxy.imageInfo.rotationDegrees
//                    )
//                )
                } catch (e: Exception) {
                    println(e.message)
                }


                try {
                    // TODO: stop passing around Activity context like this
                    val outputs = pfdHelper.pfdInference(this, imageProxy)

                    // convert imgProxy to mutable bitmap
                    val bitmap = LocalUtils(this).imageProxyToBitmap(imageProxy).copy(Bitmap.Config.RGB_565, true)

                    // call draw function to draw keypoints on previewViewSmall
                    drawPreview(outputs, bitmap)

                    // close imageProxy
                    imageProxy.close()

                } catch (e: Exception) {
                    print("Exception Occurred: ${e.message}")
                }
            }

        })

        val camera: Camera =
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
    }

    @ExperimentalGetImage
    private fun drawPreview(outputs: TensorBuffer, imgBitmap: Bitmap) {
        println(outputs.shape)

        previewView.post {
            try {
                val canvas = Canvas(imgBitmap) // create Canvas object from Bitmap

                // draw keypoints over image
                val radius = 5f // radius of the circle to draw at each keypoint

                // loop through the list of keypoints and draw a circle at each position on the canvas
                val keypoints =
                    outputs.floatArray // assuming tensorBuffer contains the list of keypoints

                val x = 100
                val y = 4
                val z = 2
                val imgSize = 512

                val reshapedArray = Array(x) {
                    Array(y) {
                        FloatArray(z)
                    }
                }

                var i = 0
                for (b in 0 until x) {
                    for (h in 0 until y) {
                        for (w in 0 until z) {
                            reshapedArray[b][h][w] = keypoints[i++] * imgSize
                        }
                    }
                }

                // final keypoint
                val finalKeypoint = reshapedArray[0]

                for (keypoint in finalKeypoint) {
                    val kx = keypoint[0].toInt()
                    val ky = keypoint[1].toInt()
//                    canvas.drawCircle(kx.toFloat(), ky.toFloat(), radius, Paint())
                }

                previewViewSmall.setImageBitmap(imgBitmap)
                previewViewSmall.invalidate()

            } catch (e: Exception) {
                print(e.message)
            }

            // set the modified Bitmap as the image for the ImageView
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        pfdHelper.destroyModel()
        harHelper.destroyModel()
    }
}

class OverlayView(context: Context, attrs: AttributeSet) : View(context, attrs) {
    private val paint = Paint()
    private val targets: MutableList<Rect> = ArrayList()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        synchronized(this) {
            for (entry in targets) {
                canvas.drawRect(entry, paint)
            }
        }
    }

    fun setTargets(sources: List<Rect>) {
        synchronized(this) {
            targets.clear()
            targets.addAll(sources)
            this.postInvalidate()
        }
    }

    init {
        val density = context.resources.displayMetrics.density
        paint.strokeWidth = 2.0f * density
        paint.color = Color.BLUE
        paint.style = Paint.Style.STROKE
    }
}