package com.example.android

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import android.widget.VideoView
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
import com.google.mlkit.vision.common.InputImage
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import android.util.Size as UtilSize


class MainActivity : AppCompatActivity(), CameraXConfig.Provider {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding
    private lateinit var previewView: PreviewView
    private lateinit var videoView: VideoView


    lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>

    private val REQUEST_CODE_PERMISSION = 101
    private val REQUIRED_PERMISSIONS: Array<String> =
        arrayOf("android.permission.CAMERA", "android.permission.READ_EXTERNAL_STORAGE")
    private val executor: Executor = Executors.newSingleThreadExecutor()
    private lateinit var harHelper: HARHelper
    private lateinit var pfdHelper: PFDHelper

    // Mediapipe vars
    private val TAG: String? = "MainActivity"
    private val BINARY_GRAPH_NAME = "pose_world_gpu.binarypb"
    private val INPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "pose_world_landmarks"
    private val NUM_HANDS = 2
    private val FLIP_FRAMES_VERTICALLY = true

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

//        AndroidAssetUtil.initializeNativeAssetManager(this);
//        val eglManager: EglManager = EglManager(this)
//        val processor: FrameProcessor = FrameProcessor(
//            this,
//            eglManager.nativeContext,
//            BINARY_GRAPH_NAME,
//            INPUT_VIDEO_STREAM_NAME,
//            OUTPUT_VIDEO_STREAM_NAME
//        )
//        processor
//            .videoSurfaceOutput
//            .setFlipY(FLIP_FRAMES_VERTICALLY)

//        processor.addPacketCallback(
//            OUTPUT_LANDMARKS_STREAM_NAME
//        ) { packet: Packet ->
//            Log.v(TAG, "Received Pose landmarks packet.")
//            try {
////                        NormalizedLandmarkList poseLandmarks = PacketGetter.getProto(packet, NormalizedLandmarkList.class);
//                val landmarksRaw = PacketGetter.getProtoBytes(packet)
////                val poseLandmarks = LandmarkList.parseFrom(landmarksRaw)
//                Log.v(
//                    TAG,
//                    "[TS:" + packet.timestamp + "] " + landmarksRaw
//                )
////                val srh: SurfaceHolder = previewDisplayView.getHolder()
////
////                  -- this line cannot Running --
////                    Canvas canvas = null;
////                    try {
////                        canvas= srh.lockCanvas();
////                        synchronized(srh){
////                            Paint paint = new Paint();
////                            paint.setColor(Color.RED);
////                            canvas.drawCircle(10.0f,10.0f,10.0f,paint);
////                        }
////                    }finally{
////                        if(canvas != null){
////                            srh.unlockCanvasAndPost(canvas);
////                        }
////                    }
//////                    processor.getVideoSurfaceOutput().setSurface(srh.getSurface());
//            } catch (exception: Exception) {
//                Log.e(TAG, "failed to get proto.", exception)
//            }
//        }

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

        imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
            //TODO: HAR implementation
//            harHelper.detectInImage(
//                InputImage.fromMediaImage(
//                    imageProxy.image!!,
//                    imageProxy.imageInfo.rotationDegrees
//                )
//            )

            // TODO: PFD implementation
            try {
                val outputs = pfdHelper.pfdInference(this, imageProxy)
            }
            catch (e:Exception) {
                print("Exception Occurred: ${e.message}")
            }

            imageProxy.close()
        })

        val camera: Camera =
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
    }

    override fun onDestroy() {
        super.onDestroy()

        pfdHelper.destroyModel()
        harHelper.destroyModel()
    }


//    private fun getPoseLandmarksDebugString(poseLandmarks: LandmarkProto.NormalizedLandmarkList): String? {
//        val poseLandmarkStr = """
//            Pose landmarks: ${poseLandmarks.landmarkCount}
//
//            """.trimIndent()
//        val poseMarkers: ArrayList<Any> = ArrayList<Any>()
//        var landmarkIndex = 0
//        for (landmark in poseLandmarks.landmarkList) {
//            val marker = PoseLandMark(landmark.x, landmark.y, landmark.z, landmark.visibility)
//            //          poseLandmarkStr += "\tLandmark ["+ landmarkIndex+ "]: ("+ (landmark.getX()*720)+ ", "+ (landmark.getY()*1280)+ ", "+ landmark.getVisibility()+ ")\n";
//            ++landmarkIndex
//            poseMarkers.add(marker)
//        }
//        Log.v(
//            TAG, """
//     ======Degree Of Position]======
//     test :${poseMarkers[16].getZ()},${poseMarkers[16].getZ()},${poseMarkers[16].getZ()}
//
//     """.trimIndent()
//        )
//        return poseLandmarkStr
//    }

}