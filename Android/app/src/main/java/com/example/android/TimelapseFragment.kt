package com.example.android

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import com.example.android.databinding.TimelapseFragmentBinding
import java.io.File

/**
 * Timelapse Fragment contains all UI logic related to timelapse displaying
 */
class TimelapseFragment : Fragment() {
    lateinit var binding: TimelapseFragmentBinding

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        binding = TimelapseFragmentBinding.inflate(inflater, container, false)

        val fileNames = requireContext().filesDir.listFiles { file ->
            file.extension == "mp4" // filter video files by extension
        }?.map { file ->
            file.name // get the name of each video file
        }?.sortedDescending()

        if (fileNames != null && fileNames.isNotEmpty()) {
            val adapter = ArrayAdapter<String>(
                requireContext(),
                R.layout.activity_list_view,
                fileNames
            )

            val listView = binding.listView
            listView.adapter = adapter

            listView.setOnItemClickListener { parent, view, position, id ->
                val videoName = listView.getItemAtPosition(position) as String
                val videoFile = File(requireContext().filesDir, videoName)

                if (videoFile.exists()) {
                    val intent = Intent(Intent.ACTION_VIEW)
                    val uri = FileProvider.getUriForFile(requireContext(), requireContext().packageName + ".provider", videoFile)
                    intent.setDataAndType(uri, "video/*")
                    intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    requireContext().startActivity(intent)
                } else {
                    Toast.makeText(context, "Video not found!", Toast.LENGTH_SHORT).show()
                }
            }
        } else {
            binding.listView.visibility = View.GONE
            binding.noContent.visibility = View.VISIBLE
        }

        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
    }
}