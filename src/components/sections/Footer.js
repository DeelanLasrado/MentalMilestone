import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-[#7fffd4] py-8">
      <div className="mx-auto max-w-7xl px-4">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-gray-900">Subscribe to Our Newsletter</h2>
          <p className="mt-2 text-gray-600">Get notified about new blog posts and upcoming community events.</p>
          <div className="mt-6 flex justify-center">
            <input
              type="email"
              placeholder="Your email address"
              className="rounded-md px-4 py-2.5 text-sm text-gray-700 focus:ring focus:ring-[#0D79F4] focus:outline-none"
            />
            <button
              className="ml-2 rounded-md bg-[#0D79F4] px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
            >
              Subscribe
            </button>
          </div>
          <div className="mt-8 text-gray-600">
            <p>
             Built with <span className="text-[#FF5733]">❤️</span> using{' '}
             <img src='/logo192.png' alt="React Logo" width="16" height="16" className="inline" />
              and
             <img src='/Tailwind.png' alt="Tailwind CSS Logo" width="16" height="16" className="inline" />
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
