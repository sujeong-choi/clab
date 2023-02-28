package com.example.android

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.annotation.NonNull
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.example.android.databinding.VideoFragmentBinding
import com.google.common.util.concurrent.ListenableFuture
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/**
 * A simple [Fragment] subclass as the default destination in the navigation.
 */
class VideoFragment : Fragment(R.layout.video_fragment), CameraXConfig.Provider {
    // view variables
    lateinit var binding: VideoFragmentBinding
    private lateinit var videoView: VideoView
    private lateinit var timeLapseView: VideoView
    private lateinit var previewView: PreviewView
    private lateinit var previewViewSmall: ImageView
    private lateinit var recordButton: Button
    private lateinit var recordText: TextView
    private lateinit var rectOverlay: RectOverlay
    private lateinit var mediaController: MediaController

    // helper and util variables
    private lateinit var commonUtils: CommonUtils
    private lateinit var harHelper: HARHelper
    private lateinit var pfdHelper: PFDHelper

    // camera variables
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var mediaMetadataRetriever: MediaMetadataRetriever
    private val executor: Executor = Executors.newSingleThreadExecutor()
    private var videoCapture: VideoCapture<Recorder>? = null
    private lateinit var videoFile: File

    // class variables
    private var isRecording: Boolean = false
    private val REQUIRED_PERMISSIONS: Array<String> =
        arrayOf(
            "android.permission.CAMERA",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.RECORD_AUDIO"
        )
    private val REQUEST_CODE_PERMISSION = 101
    private val TAG = "MAIN"

    // Create a MediaMetadataRetriever

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

    override fun getCameraXConfig(): CameraXConfig {
        return Camera2Config.defaultConfig()
    }

    @ExperimentalGetImage
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = VideoFragmentBinding.inflate(inflater, container, false)

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(requireContext(), "Failed to initialize OpenCV", Toast.LENGTH_LONG)
                .show()
        }

        // init video and preview views
        videoView = binding.videoView
        timeLapseView = binding.timelapseView
        previewView = binding.previewView
        previewViewSmall = binding.previewViewSmall
        recordButton = binding.recordButton
        recordText = binding.recordText
        rectOverlay = binding.rectOverlay

        pfdHelper = PFDHelper(requireContext())
        harHelper = HARHelper(requireContext())
        commonUtils = CommonUtils(requireContext())

        //set bg color for previewViewSmall
        previewViewSmall.setBackgroundColor(
            ContextCompat.getColor(
                requireContext(),
                android.R.color.darker_gray
            )
        )

        // allow dragging for android
        // TODO: Disable drag outside parent container area
        with(timeLapseView) {
            setOnTouchListener(object : View.OnTouchListener {
                var lastX = 0f
                var lastY = 0f

                override fun onTouch(v: View?, event: MotionEvent?): Boolean {
                    when (event?.action) {
                        MotionEvent.ACTION_DOWN -> {
                            // Save the last position of the touch
                            lastX = event.rawX
                            lastY = event.rawY

                            println("Videoview visibility: ${videoView.visibility}")
                        }
                        MotionEvent.ACTION_MOVE -> {
                            // Calculate the distance of the touch movement
                            val deltaX = event.rawX - lastX
                            val deltaY = event.rawY - lastY

                            // Get the current layout parameters of the videoView
                            val layoutParams = layoutParams as FrameLayout.LayoutParams

                            // Restrict the view from moving outside the parent layout
                            layoutParams.leftMargin += deltaX.toInt()

                            layoutParams.topMargin += deltaY.toInt()

                            lastX = event.rawX
                            lastY = event.rawY

                            // Apply the updated layout parameters
                            timeLapseView.layoutParams = layoutParams

                        }
                    }
                    return true
                }
            })
        }

        // TODO: Disable drag outside parent container area
        with(previewViewSmall) {
            setOnTouchListener(object : View.OnTouchListener {
                var lastX = 0f
                var lastY = 0f

                override fun onTouch(v: View?, event: MotionEvent?): Boolean {
                    when (event?.action) {
                        MotionEvent.ACTION_DOWN -> {
                            // Save the last position of the touch
                            lastX = event.rawX
                            lastY = event.rawY

                            println("Videoview visibility: ${videoView.visibility}")
                        }
                        MotionEvent.ACTION_MOVE -> {
                            // Calculate the distance of the touch movement
                            val deltaX = event.rawX - lastX
                            val deltaY = event.rawY - lastY

                            // Get the current layout parameters of the videoView
                            val layoutParams = layoutParams as FrameLayout.LayoutParams

                            // Restrict the view from moving outside the parent layout
                            layoutParams.leftMargin += deltaX.toInt()

                            layoutParams.topMargin += deltaY.toInt()

                            lastX = event.rawX
                            lastY = event.rawY

                            // Apply the updated layout parameters
                            previewViewSmall.layoutParams = layoutParams

                        }
                    }
                    return true
                }
            })
        }

        // init media controller

        mediaController = CustomMediaController(activity)

        mediaController.setAnchorView(videoView)
        videoView.setMediaController(mediaController)
        timeLapseView.setMediaController(mediaController)

        // live start and stop on click listeners
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

        // check camera permission and start camera
        if (!checkPermissions()) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSION
            )
        }

        cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(Runnable {
            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()
            startCamera(cameraProvider)

        }, ContextCompat.getMainExecutor(requireContext()))

        // create output video file
//        videoFile = createVideoFile()

        return binding.root
    }

    private fun checkPermissions(): Boolean {
        for (permission in REQUIRED_PERMISSIONS) {
            if (ActivityCompat.checkSelfPermission(requireContext(), permission)
                != PackageManager.PERMISSION_GRANTED
            ) {
                return false
            }
        }
        return true
    }

    private fun startRecording() {

    }

    private fun stopRecording() {

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.timelapseButton.setOnClickListener {
            findNavController().navigate(R.id.action_VideoFragment_to_TimelapseFragment)
        }

        binding.selectButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "video/*"
            startActivityForResult(intent, VIDEO_REQUEST_CODE)
        }

        binding.liveButton.setOnClickListener {
            videoView.stopPlayback()
            timeLapseView.stopPlayback()

            videoView.visibility = View.GONE
            timeLapseView.visibility = View.GONE
            previewView.visibility = View.VISIBLE
            previewViewSmall.visibility = View.VISIBLE
        }
    }

//    private fun createVideoFile(): File {
//        val videoFileName =
//            "VIDEO_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())}.mp4"
//        val storageDir = requireContext().getExternalFilesDirs(Environment.DIRECTORY_MOVIES)
//        return File(storageDir, videoFileName)
//    }

    @ExperimentalGetImage
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == VIDEO_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            val videoUri: Uri? = data?.data

            // create metadataretreiver
            try {
                mediaMetadataRetriever.setDataSource(videoUri!!.path)
            } catch (e: Exception) {
                println(e.message)
            }

            // Show VideoView and set video source
            previewView.visibility = View.GONE
            previewViewSmall.visibility = View.GONE

            videoView.visibility = View.VISIBLE
            videoView.setVideoURI(videoUri)
            videoView.requestFocus()
            videoView.start()

            // extract video frames
            val videoFrames: List<Bitmap> = commonUtils.getVideoFrames(mediaMetadataRetriever)

            // pass all video frames to pfd inference function
            // fix only pass one frame
            val output = pfdHelper.onnxInference(videoFrames[0])

            // TODO: draw keypoints on top of videoView

            // perform perspective transformation on frames
            val transformedBitmaps: List<Bitmap> =
                pfdHelper.perspectiveTransformation(videoFrames, output.keypoint.value[0])

            // save keyPoints as a timelapse
            val outputDir = requireContext().filesDir // or context.cacheDir

            val width =
                mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)!!
                    .toInt()
            val height =
                mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)!!
                    .toInt()
            val frameRate =
                mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)!!
                    .toInt()

            pfdHelper.saveVideoFromBitmaps(
                transformedBitmaps,
                outputDir.path,
                width,
                height,
                frameRate
            )

            timeLapseView.visibility = View.VISIBLE
            timeLapseView.setVideoURI(videoUri)
            timeLapseView.start()
        }
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
            ImageAnalysis.Builder().setTargetResolution(Size(1280, 720)).build()

        var lastAnalyzedTimestamp = 0L
        val predictionInterval = 10000L

        imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
            val currentTimestamp = System.currentTimeMillis()

            if (isRecording && currentTimestamp - lastAnalyzedTimestamp >= predictionInterval) {
                lastAnalyzedTimestamp = System.currentTimeMillis()

                try {
                    // initialize skeleton extractor for har
//                    harHelper.detectInImage(
//                        InputImage.fromMediaImage(
//                            imageProxy.image!!,
//                            imageProxy.imageInfo.rotationDegrees
//                        )
//                    )

//                    harHelper.harInference(InputImage.fromMediaImage(
//                        imageProxy.image!!,
//                        imageProxy.imageInfo.rotationDegrees
//                    ))
                } catch (e: Exception) {
                    println(e.message)
                }


                try {
                    // TODO: stop passing around activity context like this

                    val imgBitmap = commonUtils.imageProxyToBitmap(imageProxy)
//                    val keyPoints = pfdHelper.pfdInference(imgBitmap)
                    val pfdResult: PfdResult = pfdHelper.onnxInference(imgBitmap)

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

    @ExperimentalGetImage
    private fun drawPreview(keyPoints: KeypointType<Float>, imgBitmap: Bitmap) {
        previewView.post {
            try {
                val canvas = Canvas(imgBitmap) // create Canvas object from Bitmap

                // draw keypoints over image
                val radius = 30f

                // TODO: remove this
                // tempKeypoint definition after testing transformation
//                val tempKeypointArray = arrayOf(
//                    floatArrayOf(445f, 566f),
//                    floatArrayOf(845f, 566f),
//                    floatArrayOf(845f, 966f),
//                    floatArrayOf(445f, 966f)
//                )

                for (keypoint in keyPoints.value) {
                    val kx = keypoint[0].toInt()
                    val ky = keypoint[1].toInt()

                    val paint = Paint().apply {
                        color = Color.RED
                    }
                    canvas.drawCircle(kx.toFloat(), ky.toFloat(), radius, paint)
                }

                rectOverlay.drawOverlay(keyPoints)

                // perform perspective transformation and show image on previewViewSmall
                previewViewSmall.setImageBitmap(
                    pfdHelper.perspectiveTransformation(
                        imgBitmap,
                        keyPoints
                    )
                )
                previewViewSmall.invalidate()
            } catch (e: Exception) {
                print(e.message)
            }

            // set the modified Bitmap as the image for the ImageView
        }
    }

    companion object {
        private const val VIDEO_REQUEST_CODE = 1
    }

    override fun onDestroy() {
        super.onDestroy()

        pfdHelper.destroyModel()
        harHelper.destroyModel()
    }
}