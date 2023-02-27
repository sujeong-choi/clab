package com.example.android

import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.*
import androidx.annotation.NonNull
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import com.example.android.databinding.ActivityMainBinding
import com.google.common.util.concurrent.ListenableFuture
import com.google.mlkit.vision.common.InputImage
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import android.util.Size as UtilSize


class MainActivity : AppCompatActivity(), CameraXConfig.Provider {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var previewView: PreviewView
    private lateinit var previewViewSmall: ImageView
    private lateinit var videoView: VideoView
    private lateinit var rectOverlay: RectOverlay
    private lateinit var recordButton: Button
    private lateinit var recordText: TextView
    private var isRecording: Boolean = false
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private val REQUEST_CODE_PERMISSION = 101
    private val REQUIRED_PERMISSIONS: Array<String> =
        arrayOf(
            "android.permission.CAMERA",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.RECORD_AUDIO"
        )
    private val executor: Executor = Executors.newSingleThreadExecutor()
    private lateinit var harHelper: HARHelper
    private lateinit var pfdHelper: PFDHelper
    private lateinit var videoFile: File
    private lateinit var recorder: Recording
    private var videoCapture: VideoCapture<Recorder>? = null
    private lateinit var localUtils: LocalUtils

    private val TAG = "MAIN"


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
        harHelper = HARHelper(this)
        localUtils = LocalUtils(this)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        recordButton = findViewById(R.id.recordButton)
        recordText = findViewById(R.id.recordText)
        previewView = findViewById(R.id.previewView)
        previewViewSmall = findViewById(R.id.previewViewSmall)

        //set bg color for previewViewSmall
        previewViewSmall.setBackgroundColor(
            ContextCompat.getColor(
                this,
                android.R.color.darker_gray
            )
        )

        videoView = findViewById(R.id.videoView)
        rectOverlay = findViewById(R.id.rectOverlay)

        videoFile = createVideoFile()

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

        recordButton.setOnClickListener {
            isRecording = !isRecording
            if (isRecording) {
                recordText.visibility = View.VISIBLE
                recordButton.text = "STOP"
                previewViewSmall.visibility = View.VISIBLE

                // start video recording
                startRecording()

            } else {
                recordText.visibility = View.GONE
                recordButton.text = "RECORD"
                previewViewSmall.visibility = View.GONE

                // stop video recording
                stopRecording()
                rectOverlay.clear()
            }
        }
    }

    private fun startRecording() {

    }

    private fun stopRecording() {

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

        val recorder = Recorder.Builder()
            .setExecutor(executor)
            .build()
        videoCapture = VideoCapture.withOutput(recorder)

        val imageAnalysis: ImageAnalysis =
            ImageAnalysis.Builder().setTargetResolution(UtilSize(1280, 720)).build()

        var lastAnalyzedTimestamp = 0L
        val predictionInterval = 3000L

        imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
            val currentTimestamp = System.currentTimeMillis()

            if (isRecording && currentTimestamp - lastAnalyzedTimestamp >= predictionInterval) {
                lastAnalyzedTimestamp = System.currentTimeMillis()

                try {
                    // initialize skeleton extractor for har
                    harHelper.detectInImage(
                        InputImage.fromMediaImage(
                            imageProxy.image!!,
                            imageProxy.imageInfo.rotationDegrees
                        )
                    )

                    harHelper.harInference(InputImage.fromMediaImage(
                        imageProxy.image!!,
                        imageProxy.imageInfo.rotationDegrees
                    ))
                } catch (e: Exception) {
                    println(e.message)
                }


                try {
                    // TODO: stop passing around Activity context like this
                    val imgBitmap = localUtils.imageProxyToBitmap(imageProxy)
//                    val keyPoints = pfdHelper.pfdInference(imgBitmap)

                    // call draw function to draw keypoints on previewViewSmall
//                    drawPreview(keyPoints, imgBitmap)

                } catch (e: Exception) {
                    print("Exception Occurred: ${e.message}")
                }
            }

            // close imageProxy
            imageProxy.close()
        })

        val viewPort: ViewPort = previewView.viewPort!!
        val useCaseGroup = UseCaseGroup.Builder()
            .setViewPort(viewPort)
            .addUseCase(preview)
            .addUseCase(imageAnalysis)
            .build()

        try {
            // Bind use cases to camera
            val camera: Camera =
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    useCaseGroup,
//                    videoCapture,
                )
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun createVideoFile(): File {
        val videoFileName =
            "VIDEO_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.mp4"
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_MOVIES)
        return File(storageDir, videoFileName)
    }

    @ExperimentalGetImage
    private fun drawPreview(keyPoints: Array<FloatArray>, imgBitmap: Bitmap) {
        previewView.post {
            try {
                val canvas = Canvas(imgBitmap) // create Canvas object from Bitmap

                // draw keypoints over image
                val radius = 30f

                // TODO: remove this
                // tempKeypoint definition after testing transformation
                val tempKeypointArray = arrayOf(
                    floatArrayOf(445f, 566f),
                    floatArrayOf(845f, 566f),
                    floatArrayOf(845f, 966f),
                    floatArrayOf(445f, 966f)
                )

//                 val finalKeypoint = reshapedArray
                val finalKeypoint = tempKeypointArray

                for (keypoint in tempKeypointArray) {
                    val kx = keypoint[0].toInt()
                    val ky = keypoint[1].toInt()

                    val paint = Paint().apply {
                        color = Color.RED
                    }
                    canvas.drawCircle(kx.toFloat(), ky.toFloat(), radius, paint)
                }

                rectOverlay.drawOverlay(finalKeypoint)

                // perform perspective transformation and show image on previewViewSmall
                previewViewSmall.setImageBitmap(
                    pfdHelper.perspectiveTransformation(
                        imgBitmap,
                        finalKeypoint
                    )
                )
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