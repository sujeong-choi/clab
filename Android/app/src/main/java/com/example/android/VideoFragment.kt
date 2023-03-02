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
import android.net.Uri
import android.os.Bundle
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
import org.jetbrains.kotlinx.multik.api.d3array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.Executors


/**
 * A simple [Fragment] subclass as the default destination in the navigation.
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
    private lateinit var recordText: TextView
    private lateinit var rectOverlay: RectOverlay
    private lateinit var mediaController: MediaController
    private lateinit var loadingView: FrameLayout
    private lateinit var previewDisplayView: SurfaceView
    private lateinit var previewFrameTexture: SurfaceTexture

    // helper and util variables
    private lateinit var commonUtils: CommonUtils
    private lateinit var harHelper: HARHelper
    private lateinit var pfdHelper: PFDHelper
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var pfdSession: OrtSession
    private lateinit var harSession: OrtSession

    // camera variables
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var mediaMetadataRetriever: MediaMetadataRetriever
    private val executor: Executor = Executors.newSingleThreadExecutor()
    private var videoCapture: VideoCapture<Recorder>? = null
    private lateinit var videoFile: File

    // mediapipe variables
    private lateinit var cameraHelper: CameraXPreviewHelper
    private lateinit var applicationInfo: ApplicationInfo
    private lateinit var eglManager: EglManager
    private lateinit var converter: ExternalTextureConverter
    private lateinit var processor: FrameProcessor

    // class variables
    private var isRecording: Boolean = false
    private var isKeypointSelected: Boolean = false
    private var isCameraFacingFront: Boolean = false
    private var enableHarInference: Boolean = false
    private var globalPfdResult: PfdResult = PfdResult()
    private var globalLandmark: NormalizedLandmarkList? = null
    private var globalBitmapStore: MutableList<Bitmap> = mutableListOf()
    private var previewSize: Size? = null
    private var frameCnt = 1
    private var framesSkeleton = mk.d3array(144, 25, 3) { 0.0f }
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
        recordText = binding.recordText
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
                    if (frameCnt < 60) {//Collect Skeleton data for 60 frames or 4 seconds(4 seconds not yet implemented).
                        framesSkeleton[frameCnt] =
                            harHelper.saveSkeletonData(poseLandmarks) //save landmarks in array shape (144,25,3)
                        frameCnt++
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
                        framesSkeleton[frameCnt] = harHelper.saveSkeletonData(poseLandmarks)

                        val input = harHelper.convertSkeletonData(framesSkeleton)

                        // send to model
                        harHelper.harInference(input, harSession, ortEnv)

                        framesSkeleton =
                            mk.d3array(144, 25, 3) { 0.0f } // reinitialize landmarks array
                        frameCnt = 1

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
            isRecording = !isRecording

            if (isRecording) {
                // start video recording
                drawPreview(globalPfdResult)

                // start HAR inference
                enableHarInference = true

            } else {
                recordText.visibility = View.GONE
                recordButton.text = "RECORD"
                previewViewSmall.visibility = View.GONE

                // start HAR inference
                enableHarInference = false

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
                        bitmap?.let { it1 -> pfdHelper.pfdInference(it1, pfdSession, ortEnv) }

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

        // setup onnx model
        ortEnv = OrtEnvironment.getEnvironment()
//        val options = OrtSession.SessionOptions()
//        options.addNnapi()
//        options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.PARALLEL)
        pfdSession = ortEnv.createSession(pfdHelper.readPfdModel())
        harSession = ortEnv.createSession(harHelper.readHarModel())

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
            val output = pfdHelper.pfdInference(videoFrames[0], pfdSession, ortEnv)

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

                        // perform perspective transformation and show image on previewViewSmall

                        if (!pfdHelper.isHandInFrame(
                                bitmap!!,
                                globalPfdResult.bbox.value[0],
                                globalLandmark
                            )
                        ) {
                            requireActivity().runOnUiThread(java.lang.Runnable {
                                recordText.visibility = View.VISIBLE
                                recordButton.text = "STOP"
                                previewViewSmall.visibility = View.VISIBLE

                                val transformedBitmap = bitmap?.let {
                                    pfdHelper.perspectiveTransformation(
                                        it,
                                        keyPoints[0]
                                    )
                                }

                                previewViewSmall.setImageBitmap(transformedBitmap)
                                previewViewSmall.invalidate()
                                // hide loading spinner
                                loadingView.visibility = View.GONE
                            })
                        } else {
                            requireActivity().runOnUiThread(java.lang.Runnable {
                                Toast.makeText(
                                    requireContext(),
                                    "Hands in Painting!",
                                    Toast.LENGTH_LONG
                                )
                                    .show()

                                // hide loading spinner
                                loadingView.visibility = View.GONE
                            })
                        }


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

        pfdSession.close()
        pfdHelper.destroyModel()
        harHelper.destroyModel()
    }
}