package com.example.android

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaMetadataRetriever
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.*
import android.widget.*
import androidx.annotation.RequiresApi
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
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executor
import java.util.concurrent.Executors


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
    private var isKeypointSelected: Boolean = false
    private var isCameraFacingFront: Boolean = false
    private var enableHarInference: Boolean = false
    private var globalPfdResult: PfdResult = PfdResult()
    private var globalLandmark: NormalizedLandmarkList? = null
    private var globalBitmapStore: MutableList<Bitmap> = mutableListOf()
    private var recordingState: String = "Other"
    private var previewSize: Size? = null
    private var frameCount = 1
    private var framesSkeleton = mk.d3array(144, 25, 3) { 0.0f }
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

        // init video views
        mediaController = CustomMediaController(activity)

        mediaController.setAnchorView(videoView)
        videoView.setMediaController(mediaController)
        timeLapseView.setMediaController(mediaController)

        // create output video file
//        videoFile = createVideoFile()
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

                if (enableHarInference) {
                    if (frameCount < 60) {//Collect Skeleton data for 60 frames or 4 seconds(4 seconds not yet implemented).
                        framesSkeleton[frameCount] =
                            harHelper.saveSkeletonData(poseLandmarks) //save landmarks in array shape (144,25,3)
                        frameCount++
                    } else {
                        commonUtils.getFrameBitmap(previewDisplayView) { bitmap: Bitmap? ->
                            if (!pfdHelper.isHandInFrame(
                                    bitmap!!,
                                    globalPfdResult.bbox.value[0],
                                    globalLandmark
                                )
                            ) {
                                // update preview screen if hand isn't in frame
                                drawPreview(globalPfdResult)

                                // add bitmap to global
//                                globalBitmapStore.plus(bitmap.copy(bitmap.config, true))
                            }
                        }


                        // if hand isn't inside
                        framesSkeleton[frameCount] = harHelper.saveSkeletonData(poseLandmarks)


                        // TODO: move this inference to another thread
                        val input = harHelper.convertSkeletonData(framesSkeleton)

                        // send to model
                        val label: String = harHelper.harInference(input, harSession)

                        if (label.contains("Other")) recordingState = "Other"
                        else if (label.contains("Painting")) recordingState = "Painting"
                        else if (label.contains("Interview")) recordingState = "Interview"

                        requireActivity().runOnUiThread(java.lang.Runnable {
                            harLabel.text = label
                        })

                        clearHarSkeleton()
                    }
                }
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
        option.addNnapi()
        pfdSession = GlobalVars.ortEnv.createSession(commonUtils.readModel(ModelType.PFD), option)
        harSession = GlobalVars.ortEnv.createSession(commonUtils.readModel(ModelType.HAR), option)
    }

    private fun clearHarSkeleton() {
        framesSkeleton =
            mk.d3array(144, 25, 3) { 0.0f } // reinitialize landmarks array
        frameCount = 1
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

        binding.selectButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "video/*"
            startActivityForResult(intent, VIDEO_REQUEST_CODE)
        }

        // live button listeners
        binding.liveButton.setOnClickListener {
            videoView.stopPlayback()
            timeLapseView.stopPlayback()

            videoView.visibility = View.GONE
            timeLapseView.visibility = View.GONE
            previewView.visibility = View.VISIBLE
            previewViewSmall.visibility = View.VISIBLE
        }

        // live start and stop on click listeners
        recordButton.setOnClickListener {
            rectOverlay.clear()
            toggleRecording()

            if (isRecording) {
                // start video recording

                drawPreview(globalPfdResult)
            } else {
                // stop video recording

                // save timelapse
//                pfdHelper.saveVideoFromBitmaps(globalBitmapStore, "/", globalBitmapStore[0].width, globalBitmapStore[0].width, 30)

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

                    // for memory optimization
                    bitmap?.recycle()

                    if (pfdResult?.size != 0) {
                        globalPfdResult = pfdResult!!
                        globalPfdResult = pfdResult!!
                        isKeypointSelected = true

                        // draw keyPoints on top of previewView
                        drawKeypoints(globalPfdResult)

                    } else {
                        Toast.makeText(
                            requireContext(),
                            "Painting not detected!",
                            Toast.LENGTH_LONG
                        )
                            .show()
                    }
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
        converter.close()
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
            clearHarSkeleton()
        } else if (!isRecording) {
            recordingCircle.visibility = View.GONE
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
            clearHarSkeleton()
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
        } catch (e: Exception) {
            Log.v(TAG, "")
        }
    }

    private fun stopRecording() {
        try {
            mediaRecorder?.apply {
                stop()
//                release()
            }
//            mediaRecorder = null

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
//        val fileDir = requireContext().getExternalFilesDir(null)?.absolutePath ?: ""
        return "$fileDir/$fileName"
    }


    private fun drawPreview(pfdResult: PfdResult) {
        previewView.post {
            try {
                val keyPoints = pfdResult.keypoint.value
                val bbox = pfdResult.bbox.value

                if (keyPoints.isEmpty()) {
                    Toast.makeText(requireContext(), "Painting not detected!", Toast.LENGTH_LONG)
                        .show()
                } else {
                    commonUtils.getFrameBitmap(previewDisplayView) { bitmap: Bitmap? ->
                        requireActivity().runOnUiThread(java.lang.Runnable {
                            // show loading spinner
                            loadingView.visibility = View.VISIBLE
                        })

                        val isHandInFrame = pfdHelper.isHandInFrame(
                            bitmap!!,
                            globalPfdResult.bbox.value[0],
                            globalLandmark
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

                            })
                        } else if (isHandInFrame && recordingState == "Painting") {
                            requireActivity().runOnUiThread(java.lang.Runnable {
                                Toast.makeText(
                                    requireContext(),
                                    "Hands in Painting!",
                                    Toast.LENGTH_LONG
                                )
                                    .show()
                                toggleRecording()
                            })
                        }
                        requireActivity().runOnUiThread(java.lang.Runnable {
                            // hide loading spinner
                            loadingView.visibility = View.GONE
                        })
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
                    rectOverlay.drawKeypoints(keyPoints[0], bbox[0], enableBbox = true)
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

//        pfdSession.close()
//        harSession.close()
//        pfdHelper.destroyModel()
//        harHelper.destroyModel()
    }
}