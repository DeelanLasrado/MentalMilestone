import React, { useState } from 'react';
import AuthenticatedHeader from '../components/sections/AuthenticatedHeader';
import Footer from '../components/sections/Footer';

const Volunteer = () => {

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [reason, setReason] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Submitted Data:', { name, email, reason });
  };

  return (
    <div className="relative">
      <div className="bg-opacity-75 bg-[#7fffd4] absolute inset-0 z-[-1]"></div>
      <AuthenticatedHeader />
      <div className="mx-auto max-w-2xl py-10 sm:py-16 lg:py-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">Volunteer With Us</h1>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            We appreciate your interest in volunteering with us. Please fill out the form below.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="mt-10">
          <div className="mb-4">
            <label htmlFor="name" className="block text-gray-700 font-semibold mb-2">Name</label>
            <input
              type="text"
              id="name"
              className="w-full px-4 py-2 rounded-md border border-gray-300 focus:ring-[#0D79F4] focus:border-[#0D79F4] focus:outline-none text-gray-700"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="mb-4">
            <label htmlFor="email" className="block text-gray-700 font-semibold mb-2">Email</label>
            <input
              type="email"
              id="email"
              className="w-full px-4 py-2 rounded-md border border-gray-300 focus:ring-[#0D79F4] focus:border-[#0D79F4] focus:outline-none text-gray-700"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="mb-4">
            <label htmlFor="reason" className="block text-gray-700 font-semibold mb-2">Why Do You Want to Volunteer With Us?</label>
            <textarea
              id="reason"
              rows="4"
              className="w-full px-4 py-2 rounded-md border border-gray-300 focus:ring-[#0D79F4] focus:border-[#0D79F4] focus:outline-none text-gray-700"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            className="rounded-md bg-[#0D79F4] px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#0D79F4]"
          >
            Submit
          </button>
        </form>
      </div>
      <Footer />
    </div>
  );
};

export default Volunteer;
