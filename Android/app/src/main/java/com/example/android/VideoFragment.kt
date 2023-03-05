package com.example.android

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.media.MediaMetadataRetriever
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.*
import android.widget.*
import androidx.camera.core.ExperimentalGetImage
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
import com.google.mediapipe.components.CameraHelper.CameraFacing
import com.google.mediapipe.components.CameraHelper.OnCameraStartedListener
import com.google.mediapipe.components.CameraXPreviewHelper
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.formats.proto.LandmarkProto
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import com.google.protobuf.InvalidProtocolBufferException
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.concurrent.fixedRateTimer
import kotlin.concurrent.thread


class GlobalVars {
    companion object {
        @JvmField
        // setup onnx model
        var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
        val ortoption = OrtSession.SessionOptions()
    }
}

class VideoFragment : Fragment(R.layout.video_fragment) {
    // view variables
    private lateinit var binding: VideoFragmentBinding
    private lateinit var videoView: VideoView
    private lateinit var timeLapseView: VideoView
    private lateinit var previewView: PreviewView
    private lateinit var previewViewSmall: ImageView
    private lateinit var recordButton: Button
    private lateinit var detectButton: Button
    private lateinit var frameCounter: TextView
    private lateinit var testVad: Button
    private lateinit var recordingCircle: ImageView
    private lateinit var harLabel: TextView
    private lateinit var rectOverlay: RectOverlay
    private lateinit var mediaController: MediaController
    private lateinit var loadingView: FrameLayout
    private lateinit var previewDisplayView: SurfaceView
    private lateinit var previewFrameTexture: SurfaceTexture

    // helper and util variables
    private lateinit var commonUtils: CommonUtils
    private lateinit var harHelper: HARHelper
    private lateinit var pfdHelper: PFDHelper
    private lateinit var pfdSession: OrtSession
    private lateinit var harSession: OrtSession

    // camera variables
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var mediaMetadataRetriever: MediaMetadataRetriever
    private val executor: Executor = Executors.newSingleThreadExecutor()
    private var videoCapture: VideoCapture<Recorder>? = null
    private lateinit var videoFile: File
    private var mediaRecorder: MediaRecorder? = null

    // mediapipe variables
    private lateinit var cameraHelper: CameraXPreviewHelper
    private lateinit var applicationInfo: ApplicationInfo
    private lateinit var eglManager: EglManager
    private lateinit var converter: ExternalTextureConverter
    private lateinit var processor: FrameProcessor

    // class variables
    private var isLoading: Boolean = false
    private var isRecording: Boolean = false
    private var isNnapiAdded: Boolean = false
    private var isKeypointSelected: Boolean = false
    private var isCameraFacingFront: Boolean = false
    private var enableHarInference: Boolean = false
    private var timelapseThread: Thread? = null
    private var globalPfdResult: PfdResult = PfdResult()
    private var globalLandmark: NormalizedLandmarkList? = null
    private var globalBitmapStore: MutableList<Bitmap> = mutableListOf()
    private var recordingState: String = "No Activity"
    private var previewSize: Size? = null
    private var timelapseFps: Int = 30
    private var skeletonBuffer = ArrayList<D2Array<Float>>()
    private var initialCapture: Boolean = true
    private val REQUIRED_PERMISSIONS: Array<String> =
        arrayOf(
            "android.permission.CAMERA",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.RECORD_AUDIO"
        )
    private val REQUEST_CODE_PERMISSION = 101
    private val TAG = "VIDEOFRAG"

    private var prevSampleTime = 0L

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

    init {
        System.loadLibrary("mediapipe_jni")
        System.loadLibrary("opencv_java3")
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = VideoFragmentBinding.inflate(inflater, container, false)

        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // init OpenCV
        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(requireContext(), "Failed to initialize OpenCV", Toast.LENGTH_LONG)
                .show()
        }

        // init video and preview views
        videoView = binding.videoView
        timeLapseView = binding.timelapseView
        previewView = binding.previewView
        previewViewSmall = binding.previewViewSmall
        frameCounter = binding.frameCounter
        recordButton = binding.recordButton
        detectButton = binding.detectFrameButton
        harLabel = binding.harLabel
        testVad = binding.testVad
        recordingCircle = binding.recordingCircle
        rectOverlay = binding.rectOverlay
        loadingView = binding.loadingContainer

        // init helper and util classes
        pfdHelper = PFDHelper(requireContext())
        harHelper = HARHelper(requireContext())
        commonUtils = CommonUtils(requireContext())

        // setup fragment click listeners
        setClickListeners()

        // init mediapipe
        initMediaPipe()

        // allow dragging for android
        setupDraggablePreview()

        // disable record button until painting is detected
        recordButton.isEnabled = false

        // init video views
        mediaController = CustomMediaController(activity)

        mediaController.setAnchorView(videoView)
        videoView.setMediaController(mediaController)
        timeLapseView.setMediaController(mediaController)

        // create output video file
//        videoFile = createVideoFile()

        //create skeleton HAR timer
        val skeletonTimer =
            fixedRateTimer(name = "SkeletonTimer", initialDelay = 0L, period = 4000L) {
                getSkeleton()
            }
    }

    private fun initMediaPipe() {
        previewDisplayView = SurfaceView(requireContext())
        setupPreviewDisplayView()

        try {
            applicationInfo = requireContext().packageManager.getApplicationInfo(
                requireContext().packageName,
                PackageManager.GET_META_DATA
            )
        } catch (e: PackageManager.NameNotFoundException) {
            Log.e(TAG, "Cannot find application info: $e")
        }

        try {
            AndroidAssetUtil.initializeNativeAssetManager(requireContext())
            eglManager = EglManager(null)

            processor = FrameProcessor(
                requireContext(),
                eglManager.nativeContext,
                applicationInfo.metaData.getString("binaryGraphName"),
                applicationInfo.metaData.getString("inputVideoStreamName"),
                applicationInfo.metaData.getString("outputVideoStreamName")
            )

            processor
                .videoSurfaceOutput
                .setFlipY(true)

            processor.addPacketCallback(
                applicationInfo.metaData.getString("outputLandmarksStreamNameWorld")
            ) { packet: Packet ->
                val landmarksRaw: ByteArray = PacketGetter.getProtoBytes(packet)
                val poseLandmarks: LandmarkProto.LandmarkList =
                    LandmarkProto.LandmarkList.parseFrom(landmarksRaw)

                if (enableHarInference)
                    skeletonBuffer.add(harHelper.saveSkeletonData(poseLandmarks))
            }

            processor.addPacketCallback(
                applicationInfo.metaData.getString("outputLandmarksStreamName")
            ) { packet: Packet ->
                val landmarksRaw = PacketGetter.getProtoBytes(packet)
                try {
                    val landmarks = NormalizedLandmarkList.parseFrom(landmarksRaw)

                    if (landmarks != null) {
                        globalLandmark = landmarks
                    }
                } catch (e: InvalidProtocolBufferException) {
                    Log.e(TAG, "Couldn't Exception received - $e")
                    return@addPacketCallback
                }
            }

            setupOnnxModel()
        } catch (e: Exception) {
            e.message?.let { Log.v(TAG, it) }
        }

        converter = ExternalTextureConverter(
            eglManager.context, 2
        )
        converter.setFlipY(true)
        converter.setConsumer(processor)

        // check camera permission and start camera
        if (!checkPermissions()) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSION
            )
        } else {
            startCamera()
        }
    }

    private fun setupOnnxModel() {
        val option = GlobalVars.ortoption
        option.setIntraOpNumThreads(4)

        if (!isNnapiAdded) {
            option.addNnapi()
            isNnapiAdded = true
        }
        if (!::pfdSession.isInitialized)
            pfdSession =
                GlobalVars.ortEnv.createSession(commonUtils.readModel(ModelType.PFD), option)
        if (!::harSession.isInitialized)
            harSession = GlobalVars.ortEnv.createSession(commonUtils.readModel(ModelType.HAR))
    }

    private fun getSkeleton() {
        if (enableHarInference) {
            //for fast refreshing
            val curSkeletonBuffer = ArrayList<D2Array<Float>>()
            curSkeletonBuffer.addAll(skeletonBuffer)
            skeletonBuffer.clear()

            val input = harHelper.convertSkeletonData(curSkeletonBuffer)
            val label: String = harHelper.harInference(input, harSession)

            requireActivity().runOnUiThread(java.lang.Runnable {
                toggleHarLabel(label)
            })
        }
    }

    private fun toggleHarLabel(label: String) {
        harLabel.text = label
        if (label.contains("Painting")) {
            harLabel.setTextColor(Color.RED)
            recordingState = "Painting"
        }
        else if (label.contains("Interview")) {
            harLabel.setTextColor(Color.GREEN)
            recordingState = "Interview"
        }
        else {
            recordingState = "No Activity"
        }
    }


    private fun setupDraggablePreview() {
        //set bg color for previewViewSmall
        previewViewSmall.setBackgroundColor(
            ContextCompat.getColor(
                requireContext(),
                android.R.color.darker_gray
            )
        )

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
    }

    private fun setClickListeners() {
        // timelapse and upload button listeners
        binding.timelapseButton.setOnClickListener {
            findNavController().navigate(R.id.action_VideoFragment_to_TimelapseFragment)
        }

        testVad.setOnClickListener {
            harHelper.vadInference()
        }

        // live start and stop on click listeners
        recordButton.setOnClickListener {
            rectOverlay.clear()
            toggleRecording()

            if (isRecording) {
                // recycle bitmaps and clear globalBitmapStore to free memory
                globalBitmapStore.forEach { bitmap: Bitmap ->
                    bitmap.recycle()
                }
                globalBitmapStore.clear()

                // start video recording
                // recording thread
                timelapseThread = thread {
                    while (isRecording) {
                        commonUtils.getFrameBitmap(previewDisplayView) { bitmap: Bitmap? ->
                            // update preview screen if hand isn't in frame
                            drawPreview(globalPfdResult, globalLandmark!!, bitmap!!)
                        }
                        Thread.sleep(1000)
                    }
                }

//                drawPreview(globalPfdResult)
            } else {
                // stop video recording

                // stop timelapse thread
                timelapseThread?.join()

                // reset har label
                harLabel.text = "No Activity"

                // save timelapse
                if (globalBitmapStore.size > timelapseFps) {
                    pfdHelper.saveVideoFromBitmaps(
                        globalBitmapStore,
                        timelapseFps
                    )

                    Toast.makeText(
                        requireContext(),
                        "Timelapse saved!",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                } else {
                    Toast.makeText(
                        requireContext(),
                        "No frames detected!",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                }
            }
        }

        detectButton.setOnClickListener {
            try {
                commonUtils.getFrameBitmap(previewDisplayView) { bitmap: Bitmap? ->
                    requireActivity().runOnUiThread(java.lang.Runnable {
                        // show loading spinner
                        loadingView.visibility = View.VISIBLE
                    })

                    val pfdResult: PfdResult? =
                        bitmap?.let { it1 -> pfdHelper.pfdInference(it1, pfdSession) }

                    requireActivity().runOnUiThread(java.lang.Runnable {
                        // hide loading spinner
                        loadingView.visibility = View.GONE
                    })

                    if (pfdResult?.size != 0) {
                        globalPfdResult = pfdResult!!
                        globalPfdResult = pfdResult!!
                        isKeypointSelected = true

                        // draw keyPoints on top of previewView
                        drawKeypoints(globalPfdResult)

                        requireActivity().runOnUiThread(java.lang.Runnable {
                            recordButton.isEnabled = true
                        })

                    } else {
                        Toast.makeText(
                            requireContext(),
                            "Painting not detected!",
                            Toast.LENGTH_LONG
                        )
                            .show()
                    }

                    // for memory optimization
//                    bitmap?.recycle()
                }

            } catch (e: Exception) {
                // hide loading spinner
                loadingView.visibility = View.GONE
                e.message?.let { it1 -> Log.v("Error", it1) }
            }
        }
    }

    override fun onResume() {
        super.onResume()

        converter = ExternalTextureConverter(
            eglManager.context, 2
        )
        converter.setFlipY(true)
        converter.setConsumer(processor)

        // check camera permission and start camera
        if (!checkPermissions()) {
            ActivityCompat.requestPermissions(
                requireActivity(),
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSION
            )
        } else {
            startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        try {
            if (::converter.isInitialized) converter.close()
            previewFrameTexture.release()
//            previewFrameTexture.detachFromGLContext()
        } catch (e: java.lang.Exception) {
            e.printStackTrace()
        }
        timelapseThread?.join()
    }

    private fun toggleLoading() {
        if (isLoading) {

        } else if (!isLoading) {

        }
    }

    private fun toggleRecording() {
        isRecording = !isRecording

        if (isRecording) {
            recordingCircle.visibility = View.VISIBLE
            frameCounter.visibility = View.VISIBLE
            frameCounter.text = "0"
            harLabel.visibility = View.VISIBLE
            previewViewSmall.visibility = View.VISIBLE
            recordButton.text = "STOP"

            // disable detect button while recording
            detectButton.isEnabled = false

            // start HAR inference
            enableHarInference = true

            // start vad prediction
            harHelper.vadInference()

            startRecording()
        } else if (!isRecording) {
            recordingCircle.visibility = View.GONE
            frameCounter.visibility = View.GONE
            harLabel.visibility = View.GONE
            previewViewSmall.visibility = View.GONE
            recordButton.text = "RECORD"

            // enable detect button while recording
            detectButton.isEnabled = true

            // start HAR inference
            enableHarInference = false

            // stop vad prediction
            harHelper.vadInference()

            stopRecording()
        }
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

    private fun setupPreviewDisplayView() {
        previewDisplayView.visibility = View.GONE
        val viewGroup: ViewGroup = binding.surfaceFrame
        viewGroup.addView(previewDisplayView)

        previewDisplayView
            .holder
            .addCallback(
                object : SurfaceHolder.Callback {
                    override fun surfaceChanged(
                        holder: SurfaceHolder,
                        format: Int,
                        width: Int,
                        height: Int
                    ) {
                        val viewSize = Size(width, height)
                        previewSize = viewSize
                        val displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize)
                        val isCameraRotated = cameraHelper.isCameraRotated

                        converter.setSurfaceTextureAndAttachToGLContext(
                            previewFrameTexture,
                            if (isCameraRotated) displaySize.height else displaySize.width,
                            if (isCameraRotated) displaySize.width else displaySize.height
                        )
                    }

                    override fun surfaceCreated(holder: SurfaceHolder) {
                        processor.videoSurfaceOutput.setSurface(holder.surface)
                    }

                    override fun surfaceDestroyed(holder: SurfaceHolder) {
                        processor.videoSurfaceOutput.setSurface(null)
                    }
                })

    }

    @ExperimentalGetImage
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == VIDEO_REQUEST_CODE && resultCode == Activity.RESULT_OK) {
            val videoUri: Uri? = data?.data

            // create metadataretreiver
            try {
                mediaMetadataRetriever.setDataSource(videoUri!!.path)
            } catch (e: Exception) {
                e.message?.let { Log.v("Error", it) }
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
            val output = pfdHelper.pfdInference(videoFrames[0], pfdSession)

            // TODO: draw keypoints on top of videoView

            // perform perspective transformation on frames
//            val transformedBitmaps: List<Bitmap> =
//                pfdHelper.perspectiveTransformation(videoFrames, output.keypoint.value[0])
//
//            // save keyPoints as a timelapse
//            val outputDir = requireContext().filesDir // or context.cacheDir
//
//            val width =
//                mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)!!
//                    .toInt()
//            val height =
//                mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)!!
//                    .toInt()
//            val frameRate =
//                mediaMetadataRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)!!
//                    .toInt()
//
//            pfdHelper.saveVideoFromBitmaps(
//                transformedBitmaps,
//                outputDir.path,
//                width,
//                height,
//                frameRate
//            )

            timeLapseView.visibility = View.VISIBLE
            timeLapseView.setVideoURI(videoUri)
            timeLapseView.start()
        }
    }

    private fun startCamera() {
        cameraHelper = CameraXPreviewHelper()
        cameraHelper.setOnCameraStartedListener(
            OnCameraStartedListener { surfaceTexture ->
                previewFrameTexture = surfaceTexture!!
                // Make the display view visible to start showing the preview.
                previewDisplayView.visibility = View.VISIBLE
            })

        val cameraFacing = if (isCameraFacingFront) CameraFacing.FRONT else CameraFacing.BACK
        cameraHelper.startCamera(requireActivity(), cameraFacing,  /*unusedSurfaceTexture=*/null)

        val recorder = Recorder.Builder()
            .setExecutor(executor)
            .build()

        videoCapture = VideoCapture.withOutput(recorder)
    }

    private fun startRecording() {
        try {
            mediaRecorder = MediaRecorder(requireContext())
            mediaRecorder?.apply {
                setVideoSource(MediaRecorder.VideoSource.SURFACE)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setOutputFile(getVideoFilePath())
//                setVideoEncodingBitRate(10000000)
                setVideoFrameRate(30)
                setVideoSize(previewDisplayView.width, previewDisplayView.height)
                setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            }
            mediaRecorder?.setInputSurface(Surface(previewFrameTexture))
            mediaRecorder?.prepare()
            mediaRecorder?.start()
            mediaRecorder?.start()
        } catch (e: Exception) {
            Log.v(TAG, "")
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null

            Toast.makeText(requireContext(), "Video recording has been saved!", Toast.LENGTH_LONG)
                .show()
        } catch (e: Exception) {
            Log.v(TAG, "")
        }
    }

    private fun getVideoFilePath(): String {
        val dateFormat = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
        val currentDateAndTime: String = dateFormat.format(Date())
        val fileName = "$currentDateAndTime.mp4"
        val fileDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM)
        return "$fileDir/$fileName"
    }


    private fun drawPreview(pfdResult: PfdResult, localLandmark: NormalizedLandmarkList, bitmap: Bitmap) {
        previewView.post {
            try {
                val keyPoints = pfdResult.keypoint.value
                val bbox = pfdResult.bbox.value

                if (keyPoints.isEmpty()) {
                    Toast.makeText(requireContext(), "Painting not detected!", Toast.LENGTH_LONG)
                        .show()
                } else {
                    val isHandInFrame = pfdHelper.isHandInFrame(
                        bitmap,
                        globalPfdResult.bbox.value[0],
                        localLandmark
                    )

                    // perform perspective transformation and show image on previewViewSmall
                    if (!isHandInFrame && (recordingState == "Painting" || initialCapture)) {
                        requireActivity().runOnUiThread(java.lang.Runnable {
                            val transformedBitmap = bitmap.let {
                                pfdHelper.perspectiveTransformation(
                                    it,
                                    keyPoints[0]
                                )
                            }

                            previewViewSmall.setImageBitmap(transformedBitmap)
                            previewViewSmall.invalidate()

                            initialCapture = false

                            // add bitmap to global
                            globalBitmapStore.add(transformedBitmap)

                            val plusOne: String =
                                (frameCounter.text.toString().toInt() + 1).toString()
                            frameCounter.text = plusOne
                        })
                    } else if (isHandInFrame && recordingState == "Painting") {
                        // Do something when hand is detected inside frame during painting
                    }

                }
            } catch (e: Exception) {
                e.message?.let { Log.v("Error", it) }
            }

            // set the modified Bitmap as the image for the ImageView
        }
    }

    private fun drawKeypoints(pfdResult: PfdResult) {
        previewView.post {
            try {
                val keyPoints = pfdResult.keypoint.value
                val bbox = pfdResult.bbox.value

                if (keyPoints.isEmpty()) {
                    Toast.makeText(requireContext(), "Painting not detected!", Toast.LENGTH_LONG)
                        .show()
                } else {
                    rectOverlay.drawKeypoints(keyPoints[0], bbox[0], enableBbox = false)
                }
            } catch (e: Exception) {
                e.message?.let { Log.v("Error", it) }
            }

            // set the modified Bitmap as the image for the ImageView
        }
    }

    companion object {
        private const val VIDEO_REQUEST_CODE = 1
    }

    override fun onDestroy() {
        super.onDestroy()

//        converter.close()

        // Hide preview display until we re-open the camera again.
//        previewDisplayView.visibility = View.GONE
        pfdHelper.destroyModel()
        harHelper.destroyModel()
    }
}