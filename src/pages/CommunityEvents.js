import React from 'react';
import AuthenticatedHeader from '../components/sections/AuthenticatedHeader';
import Footer from '../components/sections/Footer';

const Card = ({ imageSrc, title, description, buttonText }) => {
  return (
    <div className="bg-[#12e0b738] p-6 rounded-md shadow-md m-4">
      <div className="relative">
        <img src={imageSrc} alt={title} className="h-full w-full object-cover object-center" />
      </div>
      <div className="p-4">
        <h2 className="text-2xl font-bold text-gray-700">{title}</h2>
        <p className="text-gray-700 mt-2">{description}</p>
        <div className="mt-4">
          <button
            className="rounded-md bg-[#0D79F4] px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-100 hover:text-[#0D79F4] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            {buttonText}
          </button>
        </div>
      </div>
    </div>
  );
};

const CommunityEvents = () => {
  return (
    <div className="relative">
      <div className="bg-opacity-75 bg-[#7fffd4] absolute inset-0 z-[-1]"></div>
      <AuthenticatedHeader />
      <div>
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">Community Events</h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Discover our upcoming community events and join us for meaningful experiences.
          </p>
        </div>
      </div>
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 sm:py-16 lg:py-20">
        <div className="grid grid-cols-1 gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 xl:gap-x-8">

          <Card
            imageSrc="https://images.unsplash.com/photo-1491438590914-bc09fcaaf77a?auto=format&fit=crop&q=80&w=2070&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
            title="Community Picnic"
            description="Join us this Sunday at the Central Park, University of Cincinnati for a community picnic! This event, organized for teenagers, promises a day of connection and fun. Enjoy games, music, and a chance to share your experiences."
            buttonText="Register"
          />

          <Card
            imageSrc="https://images.unsplash.com/photo-1564121211835-e88c852648ab?auto=format&fit=crop&q=80&w=2070&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
            title="Mental Health Awareness Workshop"
            description="Calling all volunteers! The Community Center is hosting a workshop on mental health awareness and support. Join us on November 6th to learn how to be a pillar of support for members in our community."
            buttonText="Register"
          />

          <Card
            imageSrc="https://images.unsplash.com/photo-1521225753516-46438a76f25a?auto=format&fit=crop&q=80&w=1974&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
            title="Poetry Night at the VOC Auditorium"
            description="Get ready for an exciting evening at the VOC Auditorium! Our Poetry Night is happening on November 12th. Explore your creativity, share your stories, and connect with a vibrant community of fellow artists."
            buttonText="Register"
          />
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default CommunityEvents;
