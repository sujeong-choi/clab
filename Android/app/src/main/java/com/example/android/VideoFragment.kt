package com.example.android

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.*
import android.widget.*
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.example.android.databinding.VideoFragmentBinding
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
import java.nio.ByteBuffer
import kotlin.concurrent.thread
import kotlin.math.abs


class GlobalVars {
    companion object {
        @JvmField
        // setup global onnx model environment
        var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    }
}

interface GlobalLandmarkType {
    var landmark: NormalizedLandmarkList?
    var timestamp: Long
}

/**
 * Video Fragment contains all UI logic related to video processing, management of model files and display and using API
 */
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
    private var pfdSession: OrtSession? = null
    private var harSession: OrtSession? = null
    private lateinit var onnxOptions: OrtSession.SessionOptions

    // mediapipe variables
    private var cameraHelper: CameraXPreviewHelper? = null
    private lateinit var applicationInfo: ApplicationInfo
    private lateinit var eglManager: EglManager
    private var converter: ExternalTextureConverter? = null
    private lateinit var processor: FrameProcessor
    private var frameSize: Size = Size(0, 0)

    // class variables
    private var isRecording: Boolean = false
    private var isKeypointSelected: Boolean = false
    private var isCameraFacingFront: Boolean = false
    private var enableHarInference: Boolean = false
    private var timelapseThread: Thread? = null
    private var harSkeletonThread: Thread? = null
    private var recordThread: Thread? = null
    private var detectThread: Thread? = null
    private var saveVideoThread: Thread? = null
    private var globalPfdResult: PfdResult = PfdResult()
    private var globalLandmark: MutableList<GlobalLandmarkType> = mutableListOf()
    private var currentFrame: Bitmap? = null
    private var currentFrameTimestamp: Long = 0L
    private lateinit var timelapseId: String
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

    init {
        // load opencv3 and mediapipe libraries
        System.loadLibrary("mediapipe_jni")
        System.loadLibrary("opencv_java3")
    }


    // Android Lifecycle Methods
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = VideoFragmentBinding.inflate(inflater, container, false)

        return binding.root
    }

    // setup UI bindings and init mediapipe + opencv + model here
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
        setupDraggableKeypoints()

        // disable record button until painting is detected
        recordButton.isEnabled = false

        // init video views
        mediaController = CustomMediaController(activity)

        mediaController.setAnchorView(videoView)
        videoView.setMediaController(mediaController)
        timeLapseView.setMediaController(mediaController)
    }


    // override on resume to resume unused resources
    override fun onResume() {
        super.onResume()

        converter = ExternalTextureConverter(
            eglManager.context, 2
        )
        converter?.setFlipY(true)
        converter?.setConsumer(processor)

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

    // override on pause to freeup unused resources
    override fun onPause() {
        super.onPause()
        try {
            if (converter != null) {
                converter?.close()
                converter = null
            }
            previewFrameTexture.release()
        } catch (e: java.lang.Exception) {
            e.printStackTrace()
        }

        // Hide preview display until we re-open the camera again.
        previewDisplayView.visibility = View.GONE

        // stop timelapse thread
        timelapseThread?.join()
    }

    override fun onDestroy() {
        super.onDestroy()
        // Hide preview display until we re-open the camera again.
        pfdHelper.destroyModel()
        harHelper.destroyModel()

        // destroy threads
        timelapseThread?.join()
        harSkeletonThread?.join()
        recordThread?.join()
        detectThread?.join()
        saveVideoThread?.join()
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

    // init media pipe and set variables according to official docs
    private fun initMediaPipe() {
        if (::processor.isInitialized) {
            processor.close()
        }

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
                if (enableHarInference)
                    skeletonBuffer.add(
                        harHelper.saveSkeletonData(
                            LandmarkProto.LandmarkList.parseFrom(
                                PacketGetter.getProtoBytes(packet)
                            )
                        )
                    )
            }

            processor.addPacketCallback(
                applicationInfo.metaData.getString("outputLandmarksStreamName")
            ) { packet: Packet ->
                val landmarksRaw = PacketGetter.getProtoBytes(packet)
                try {
                    if (NormalizedLandmarkList.parseFrom(landmarksRaw) != null) {
                        if (globalLandmark.size >= 120)
                            globalLandmark.clear()

                        globalLandmark.add(object : GlobalLandmarkType {
                            override var landmark: NormalizedLandmarkList? =
                                NormalizedLandmarkList.parseFrom(landmarksRaw)
                            override var timestamp: Long = packet.timestamp
                        })
                    }
                } catch (e: InvalidProtocolBufferException) {
                    Log.e(TAG, "Couldn't Exception received - $e")
                    return@addPacketCallback
                }
            }

            processor.addPacketCallback(
                applicationInfo.metaData.getString("outputImageStreamName")
            ) { packet ->
                val width = PacketGetter.getImageWidth(packet)
                val height = PacketGetter.getImageHeight(packet)
                val buffer: ByteBuffer = ByteBuffer.allocateDirect(width * height * 4)
                val isConverted: Boolean = PacketGetter.getImageData(packet, buffer)

                if (isConverted) {
                    val bitmap: Bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    bitmap.copyPixelsFromBuffer(buffer)

                    // do something with bitmap
//                    if(!initialCapture) currentFrame?.recycle()
                    currentFrame = bitmap
                    currentFrameTimestamp = packet.timestamp
                }
            }

            setupOnnxModel()
        } catch (e: Exception) {
            e.message?.let { Log.v(TAG, it) }
        }

        // create mediapipe converter (refer )
        converter = ExternalTextureConverter(
            eglManager.context, 2
        )
        converter?.setFlipY(true)
        converter?.setConsumer(processor)

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

    // setup onnx model using global onnx env
    private fun setupOnnxModel() {
        val env = GlobalVars.ortEnv
        onnxOptions = OrtSession.SessionOptions()
        onnxOptions.setIntraOpNumThreads(2)
        onnxOptions.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
    }

    // get skeletons for HAR processing
    private fun getSkeleton() {
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

    // toggle HAR label
    private fun toggleHarLabel(label: String) {
        harLabel.text = label
        if (label.contains("Painting")) {
            harLabel.setTextColor(Color.RED)
            recordingState = "Painting"
        } else if (label.contains("Interview")) {
            harLabel.setTextColor(Color.GREEN)
            recordingState = "Interview"
        } else {
            recordingState = "No Activity"
        }
    }


    // UI Config functions
    // setup draggable preview for timelapse frames
    private fun setupDraggablePreview() {
        //set bg color for previewViewSmall
        previewViewSmall.setBackgroundColor(
            ContextCompat.getColor(
                requireContext(),
                android.R.color.darker_gray
            )
        )

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

    // setup all onclick listeners
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
            isRecording = !isRecording

            if (isRecording) {
                requireActivity().runOnUiThread(java.lang.Runnable {
                    // show loading spinner
                    loadingView.visibility = View.VISIBLE
                })

                recordThread = thread {
                    // close PFD session for memory management
                    pfdSession?.close()
                    pfdSession = null

                    // set timelapse id to name file later
                    timelapseId = pfdHelper.getTimelapseId()

                    System.gc()

                    // init har model
                    if (harSession == null) {
                        val env = GlobalVars.ortEnv
                        harSession =
                            env.createSession(commonUtils.readModel(ModelType.HAR))
                    }

                    // start HAR inference once HAR model is initialized
                    enableHarInference = true

                    // start vad prediction
                    harHelper.vadInference()

                    // update ui for recording
                    requireActivity().runOnUiThread(java.lang.Runnable {
                        // prepare ui for recording
                        recordingCircle.visibility = View.VISIBLE
                        frameCounter.visibility = View.VISIBLE
                        frameCounter.text = "0"
                        harLabel.visibility = View.VISIBLE
                        previewViewSmall.visibility = View.VISIBLE
                        recordButton.text = "STOP"

                        // disable detect button while recording
                        detectButton.isEnabled = false
                    })

                    // thread that captures timelapses
                    timelapseThread = thread {
                        while (isRecording) {
                            val landmarkList = globalLandmark.toList()
                            var localLandmark: LandmarkProto.NormalizedLandmarkList? = null
                            var minDiff = 50L
                            val localFrame = currentFrame
                            val localFrameTimestamp = currentFrameTimestamp

                            // find the landmark closest in time to the current bitmap
                            // iterate through each landmarklist
                            for (i in landmarkList.indices) {
                                val l = landmarkList[i]
                                val diff = abs(localFrameTimestamp - l.timestamp)

                                if (diff < minDiff) {
                                    minDiff = diff
                                    localLandmark = l.landmark!!
                                }
                            }

                            Log.v(TAG, "Last skeleton timestamp: ${(landmarkList.last().timestamp / 1000000).toInt()}")
                            Log.v(TAG, "Frame timestamp: ${(localFrameTimestamp / 1000000).toInt()}")

                            // update preview screen if hand isn't in frame
                            drawPreview(globalPfdResult, localLandmark, localFrame)

                            Thread.sleep(1000)
                        }
                    }

                    // thread that captures skeletons for HAR
                    harSkeletonThread = thread {
                        while (enableHarInference) {
                            getSkeleton()
                            Thread.sleep(4000)
                        }
                    }

                    requireActivity().runOnUiThread(java.lang.Runnable {
                        // show loading spinner
                        loadingView.visibility = View.GONE
                    })

                }
            } else {
                requireActivity().runOnUiThread(java.lang.Runnable {
                    // show loading spinner
                    loadingView.visibility = View.VISIBLE

                    recordingCircle.visibility = View.GONE
                    frameCounter.visibility = View.GONE
                    harLabel.visibility = View.GONE
                    previewViewSmall.visibility = View.GONE
                    recordButton.text = "RECORD"

                    // enable detect button while recording
                    detectButton.isEnabled = true
                })

                // stop HAR inference
                enableHarInference = false

                // stop har and vad prediction
                harSession?.close()
                harSession = null
                harHelper.vadInference()

                // stop timelapse thread
                timelapseThread?.join()

                // stop harSkeleton thread
                harSkeletonThread?.join()

                // save timelapse
                val totalFrames: Int = frameCounter.text.toString().toInt() + 1

                var isSaved = false

                saveVideoThread = thread {
                    isSaved =
                        pfdHelper.saveVideoFromBitmaps(timelapseFps, timelapseId, totalFrames)
                }

                saveVideoThread?.join()

                if (isSaved) {
                    Toast.makeText(
                        requireContext(),
                        "Timelapse saved!",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                } else {
                    Toast.makeText(
                        requireContext(),
                        "Not enough frames detected!",
                        Toast.LENGTH_SHORT
                    )
                        .show()
                }

                requireActivity().runOnUiThread(java.lang.Runnable {
                    // show loading spinner
                    loadingView.visibility = View.GONE
                })
            }
        }

        // detect button click listener
        detectButton.setOnClickListener {
            try {
                requireActivity().runOnUiThread(java.lang.Runnable {
                    // show loading spinner
                    loadingView.visibility = View.VISIBLE
                })

                rectOverlay.clear()

                detectThread = thread {
                    if (pfdSession == null) {
                        val env = GlobalVars.ortEnv
                        val modelFile = commonUtils.readModel(ModelType.PFD)
                        pfdSession =
                            env.createSession(modelFile)
                    }

                    val pfdResult: PfdResult? =
                        currentFrame?.let { it1 -> pfdHelper.pfdInference(it1, pfdSession) }

                    requireActivity().runOnUiThread(java.lang.Runnable {
                        // hide loading spinner
                        loadingView.visibility = View.GONE
                    })

                    if (pfdResult?.size != 0) {
                        globalPfdResult = pfdResult!!
                        isKeypointSelected = true

                        // draw keyPoints on top of previewView
                        drawKeypoints(globalPfdResult)

                        requireActivity().runOnUiThread(java.lang.Runnable {
                            recordButton.isEnabled = true
                        })

                    } else {
                        requireActivity().runOnUiThread {
                            Toast.makeText(
                                requireContext(),
                                "Painting not detected!",
                                Toast.LENGTH_LONG
                            )
                                .show()
                        }
                    }
                }
            } catch (e: Exception) {
                // hide loading spinner
                loadingView.visibility = View.GONE
                e.message?.let { it1 -> Log.v("Error", it1) }
            }
        }
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
                        val displaySize = cameraHelper!!.computeDisplaySizeFromViewSize(viewSize)
                        val isCameraRotated = cameraHelper!!.isCameraRotated

                        frameSize = Size(
                            if (isCameraRotated) displaySize.height else displaySize.width,
                            if (isCameraRotated) displaySize.width else displaySize.height
                        )

                        val w = binding.surfaceFrame.width
                        val h = binding.surfaceFrame.height
                        rectOverlay.setScreenSize(Size(w, h))

                        converter?.setSurfaceTextureAndAttachToGLContext(
                            previewFrameTexture,
                            frameSize.width,
                            frameSize.height
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

    private fun startCamera() {
        try {
            cameraHelper = CameraXPreviewHelper()
            previewFrameTexture = converter?.surfaceTexture!!

            cameraHelper!!.setOnCameraStartedListener(
                OnCameraStartedListener { surfaceTexture ->
                    previewFrameTexture = surfaceTexture!!
                    // Make the display view visible to start showing the preview.
                    previewDisplayView.visibility = View.VISIBLE
                })

            val cameraFacing = if (isCameraFacingFront) CameraFacing.FRONT else CameraFacing.BACK
            cameraHelper!!.startCamera(
                requireActivity(),
                cameraFacing,  /*unusedSurfaceTexture=*/
                null
            )
        } catch (e: Exception) {
            e.message?.let { Log.v(TAG, it) }
        }
    }

    private fun drawPreview(
        pfdResult: PfdResult,
        localLandmark: LandmarkProto.NormalizedLandmarkList?,
        bitmap: Bitmap?
    ) {
        previewView.post {
            try {
                val keyPoints = pfdResult.keypoint.value
                val bbox = pfdResult.bbox.value

                if (keyPoints.isEmpty() || bitmap == null) {
                    Toast.makeText(requireContext(), "Painting not detected!", Toast.LENGTH_LONG)
                        .show()
                } else {
                    val isHandInFrame = pfdHelper.isHandInFrame(
                        bitmap,
                        pfdResult.bbox.value[0],
                        localLandmark,
                        isStrict = true
                    )

                    Log.v(
                        TAG,
                        "Is hand in frame: $isHandInFrame, didn't find skeleton in time: ${localLandmark == null}"
                    )

                    // perform perspective transformation and show image on previewViewSmall
//                    if (initialCapture || (!isHandInFrame && recordingState == "Painting")) {
                    if (initialCapture || !isHandInFrame) {
                        requireActivity().runOnUiThread(java.lang.Runnable {
                            val transformedBitmap = bitmap.let {
                                pfdHelper.perspectiveTransformation(
                                    it,
                                    keyPoints[0]
                                )
                            }

                            initialCapture = false

                            previewViewSmall.setImageBitmap(transformedBitmap)
                            previewViewSmall.invalidate()

                            val plusOne: Int = frameCounter.text.toString().toInt() + 1
                            frameCounter.text = plusOne.toString()

                            // save bitmap to internal storage to compile into timelapse later
                            pfdHelper.saveBitmapToInternalStorage(
                                transformedBitmap,
                                timelapseId,
                                plusOne
                            )

                            bitmap?.recycle()
                        })
                    } else if (isHandInFrame && recordingState == "Painting") {
                        // Do something when hand is detected inside frame during painting
                    }

                }
            } catch (e: Exception) {
                e.message?.let { Log.v("Error", it) }
            }
        }
    }

    private fun drawKeypoints(pfdResult: PfdResult, radius: Int = 15, enableBbox: Boolean = false) {
        previewView.post {
            try {
                val keyPoints = pfdResult.keypoint.value
                val bbox = pfdResult.bbox.value

                if (keyPoints.isEmpty()) {
                    Toast.makeText(requireContext(), "Painting not detected!", Toast.LENGTH_LONG)
                        .show()
                } else {
                    rectOverlay.drawKeypoints(keyPoints[0], bbox[0], enableBbox, radius)
                }
            } catch (e: Exception) {
                e.message?.let { Log.v("Error", it) }
            }
        }
    }

    private fun setupDraggableKeypoints() {
        with(rectOverlay) {
            // -1 because no point has been selected
            var keypointIndex = -1
            val threshold = 15

            setOnTouchListener(object : View.OnTouchListener {
                override fun onTouch(v: View?, event: MotionEvent?): Boolean {

                    when (event?.action) {
                        MotionEvent.ACTION_DOWN -> {
                            if (globalPfdResult.keypoint.value.isNotEmpty() && !enableHarInference) {
                                val keypoints = globalPfdResult.keypoint.value[0]

                                // iterate through keypoints to see if any of them are within the on touch threshold
                                for (i in keypoints.indices) {
                                    // check if (prevX, prevY) and threshold are within the keypoint (x, y)
                                    if (keypoints[i][0] in event.x - threshold..event.x + threshold && keypoints[i][1] in event.y - threshold..event.y + threshold) {
                                        keypointIndex = i
                                        break
                                    }
                                }
                            }
                        }
                        MotionEvent.ACTION_MOVE -> {
                            if (keypointIndex != -1) {
                                globalPfdResult.keypoint.value[0][keypointIndex][0] = event.x
                                globalPfdResult.keypoint.value[0][keypointIndex][1] = event.y

                                drawKeypoints(globalPfdResult, 20)
                            }
                        }
                        MotionEvent.ACTION_UP -> {
                            // reset keypoint index when touch event starts
                            keypointIndex = -1

                            // add to check if the global
                            if (globalPfdResult.keypoint.value.isNotEmpty() && !enableHarInference) {
                                // disable isDragEvent on keypoints
                                drawKeypoints(globalPfdResult)
                            }
                        }
                    }
                    return true
                }
            })
        }
    }
}