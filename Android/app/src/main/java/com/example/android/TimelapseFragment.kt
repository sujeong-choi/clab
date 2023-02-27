package com.example.android

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.navigation.fragment.findNavController
import com.example.android.databinding.TimelapseFragmentBinding

/**
 * A simple [Fragment] subclass as the second destination in the navigation.
 */
class TimelapseFragment : Fragment() {
    lateinit var binding: TimelapseFragmentBinding

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        binding = TimelapseFragmentBinding.inflate(inflater, container, false)
        return binding.root

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        binding.prevButton.setOnClickListener{
            findNavController().navigate(R.id.action_TimelapseFragment_to_VideoFragment)
        }

        //TODO: find and display previous timelapse videos
    }
}