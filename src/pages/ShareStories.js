import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

const StoryCard = () => {
  const [userData, setUserData] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const fetchUserData = async () => {
    try {
      const response = await fetch('https://randomuser.me/api/');
      const data = await response.json();

      const user = data.results[0];
      setUserData({
        name: `${user.name.first} ${user.name.last}`,
        profilePicture: user.picture.large,
      });

      setIsLoading(false);
    } catch (error) {
      console.error('Error fetching user data:', error);
    }
  };

  useEffect(() => {
    fetchUserData();
  }, []);

  return (
    <article className="flex max-w-xl flex-col items-start justify-between" style={{ backgroundColor: "#07a5f738", padding: "30px", borderRadius: "10px" }}>
    <div className="flex items-center gap-x-6 text-xs">
      <time dateTime="2020-03-16" className="text-gray-500">Nov 5, 2023</time>
      <Link to="/adult/ShareStories/story" className="relative z-10 font-medium text-gray-700 text-xl">Life-changing App</Link>
    </div>
    <div className="group relative">
      <h3 className="mt-3 text-lg font-semibold leading-6 text-gray-900 group-hover:text-gray-600">
        <Link to="/adult/ShareStories/story" >
          <span className="absolute inset-0"></span>
        </Link>
      </h3>
      <p className="mt-5 line-clamp-3 text-sm leading-6 text-gray-900">This app transformed my daily routine. I love its sentiment analysis and calendar features. The alert system is a game-changer, ensuring my mental well-being. Volunteering and community events? A great addition!</p>
    </div>
    <div className="relative mt-8 flex items-center gap-x-4">
      <img src={isLoading ? 'loading.gif' : userData.profilePicture} alt="" className="h-10 w-10 rounded-full bg-gray-50" />
      <div className="text-sm leading-6">
        <p className="font-semibold text-gray-900">
          <Link to="/ShareStories#">
            <span className="absolute inset-0"></span>
            {isLoading ? 'Loading...' : userData.name}
          </Link>
        </p>
        <p className="text-gray-600">Community Member</p>
      </div>
    </div>
  </article>
  )
}

const ShareStories = () => {
  return (
    <div className="py-24 sm:py-32" style={{ backgroundColor: "#0d87f430" }}>
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl" style={{ textAlign: "center" }}>Stories from our vibrant Community</h2>
          <p className="mt-2 text-lg leading-8 text-gray-600" style={{ textAlign: "center" }}>Stories for you, by you.</p>
        </div>

        <div className="mx-auto mt-10 grid max-w-2xl grid-cols-1 gap-x-8 gap-y-16 border-t border-gray-300 pt-10 sm:mt-16 sm:pt-16 lg:mx-0 lg:max-w-none lg:grid-cols-3">
          <StoryCard />
          <StoryCard />
          <StoryCard />
        </div>
      </div>
    </div>
  );
}

export default ShareStories
