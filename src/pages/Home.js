import React from 'react';
import AuthenticatedHeader from '../components/sections/AuthenticatedHeader';
import Footer from '../components/sections/Footer';
import { Link } from 'react-router-dom';
import { BiCheckCircle, BiCalendar } from 'react-icons/bi';
import { motion } from 'framer-motion';

const Home = () => {
  const missionVariants = {
    hidden: { opacity: 0, x: -100 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.8,
      },
    },
  };

  const visionVariants = {
    hidden: { opacity: 0, x: 100 },
    visible: {
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.8,
      },
    },
  };

  return (
    <div className="relative">
      <div className="bg-opacity-75 bg-[#7fffd4] absolute inset-0 z-[-1]"></div>
      <AuthenticatedHeader />
      <div className="mx-auto max-w-2xl py-10 sm:py-16 lg:py-8">
        <div className="hidden sm:mb-8 sm:flex sm:justify-center">
          <div className="relative rounded-full px-3 py-1 text-sm leading-6 text-gray-600 ring-1 ring-gray-900/10 hover:ring-gray-900/20">
            Mental Milestone: Your Path to Well-Being.
            <Link to="/features" className="font-semibold text-[#0D79F4]">
              <span className="absolute inset-0" aria-hidden="true"></span> Read more <span aria-hidden="true">&rarr;</span>
            </Link>
          </div>
        </div>
      </div>
      <div>
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">Empower Your Mental Well-Being</h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Mental Milestone is your partner for achieving mental well-being, providing tools and insights to support your journey.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              to="/about-us"
              className="rounded-md bg-[#0D79F4] px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              Learn More
            </Link>
            <Link to="/team" className="text-sm font-semibold leading-6 text-gray-900">
              Meet Our Team <span aria-hidden="true">→</span>
            </Link>
          </div>
        </div>
      </div>
      <div className="flex justify-center mt-4">
        <motion.div
          className="w-1/2 bg-[#7fffd4] border-green-500 border rounded-md shadow-md p-6 m-4"
          variants={missionVariants}
          initial="hidden"
          animate="visible"
        >
          <div className="flex items-center justify-center">
            <BiCheckCircle className="text-4xl text-[#57C7FF]" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mt-4">Our Mission</h2>
          <p className="text-gray-900 mt-2">
            Our mission is to provide comprehensive support and resources to individuals on their mental well-being journey, empowering them to lead fulfilling lives.
          </p>
        </motion.div>

        <motion.div
          className="w-1/2 bg-[#7fffd4] border-green-500 border rounded-md shadow-md p-6 m-4"
          variants={visionVariants}
          initial="hidden"
          animate="visible"
        >
          <div className="flex items-center justify-center">
            <BiCalendar className="text-4xl text-[#FF9335]" />
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mt-4">Our Vision</h2>
          <p className="text-gray-900 mt-2">
            Our vision is a world where individuals are empowered to take control of their mental well-being, where they have access to the tools and knowledge needed to thrive.
          </p>
        </motion.div>
      </div>
      <Footer />
    </div>
  );
};

export default Home;
