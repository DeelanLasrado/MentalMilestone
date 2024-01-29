import React, { useState } from 'react';
import AuthenticatedHeader from '../components/sections/AuthenticatedHeader';

const Videos = () => {
  const videoData = [
    {
      title: 'The Impact of Self Talk',
      url: 'https://www.youtube.com/embed/iNyUmbmQQZg?si=ejnot6BS4_o-W_bT',
    },
    {
      title: 'Are You Emotionally Mature?',
      url: 'https://www.youtube.com/embed/Uew5BbvmLks?si=fDEY2ME2OMwCTGim',
    },
    {
      title: 'Why you feel anxious socializing (and what to do about it) - Fallon Goodman',
      url: 'https://www.youtube.com/embed/RqAOrr_FPxo?si=Vd_C8TWjAZzqLNqf',
    },
    {
      title: 'How Love Changes Your Brain?',
      url: 'https://www.youtube.com/embed/f_OPjYQovAE',
    },
    {
      title: 'How Friendship Changes Your Brain?',
      url: 'https://www.youtube.com/embed/YmVpwXH4jhA?si=I0RZ_EDem7e7or8j',
    },
  ];

  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);

  const handleNextVideo = () => {
    setCurrentVideoIndex((prevIndex) =>
      prevIndex < videoData.length - 1 ? prevIndex + 1 : 0
    );
  };

  const handlePreviousVideo = () => {
    setCurrentVideoIndex((prevIndex) =>
      prevIndex > 0 ? prevIndex - 1 : videoData.length - 1
    );
  };

  const currentVideo = videoData[currentVideoIndex];

  return (
    <div className="relative bg-[#7fffd4]">
      <AuthenticatedHeader />
      <div className="video-carousel-container mx-auto max-w-2xl py-10 sm:py-16 lg:py-8">
        <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
          {currentVideo.title}
        </h1>
        <div className="video-container mt-6 aspect-w-16 aspect-h-9">
          <iframe
            src={currentVideo.url}
            title={currentVideo.title}
            className="w-full h-[400px] lg:h-[500px] rounded-lg"
            frameBorder="0"
            allowFullScreen
          ></iframe>
        </div>
        <div className="mt-6 flex justify-between video-controls">
          <button
            onClick={handlePreviousVideo}
            className="rounded-md bg-[#0D79F4] px-3.5 py-2.5 text-sm font-semibold text-white hover-bg-indigo-100 hover-text-[#0D79F4] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            Previous
          </button>
          <button
            onClick={handleNextVideo}
            className="rounded-md bg-[#0D79F4] px-3.5 py-2.5 text-sm font-semibold text-white hover-bg-indigo-100 hover-text-[#0D79F4] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            Next
          </button>
        </div>
        <p className="text-gray-600 text-sm mt-4">
          Source: <a href="https://ed.ted.com/lessons?category=emotional-health" className="text-[#0D79F4]" target="_blank" rel="noopener noreferrer">https://ed.ted.com/lessons?category=emotional-health</a>
        </p>
      </div>
    </div>
  );
};

export default Videos;
