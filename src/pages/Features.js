import React from 'react';

const Features = () => {
  const features = [
    {
      title: 'Journaling',
      description: 'Users can write journal entries to express their thoughts and feelings.',
    },
    {
      title: 'Sentiment Analysis',
      description: 'The application analyzes the sentiment of user entries using AI LLM to understand their emotional state.',
    },
    {
      title: 'Sentiment-Based Calendar',
      description: 'A React calendar app marks each day as red, green, or blue based on the user\'s sentiment, providing a visual overview of their emotional trends.',
    },
    {
      title: 'Alert System',
      description: 'The system sends out emails to the user\'s near and dear ones if there are three consecutive "red" or negative sentiment days, ensuring timely support.',
    },
    {
      title: 'Educational Videos',
      description: 'Curated Educational Videos for Informed Self-Improvement and Mental Well-Being.',
    },
    {
      title: 'Volunteering',
      description: 'Users can engage in volunteering opportunities and activities related to mental health.',
    },
    {
      title: 'Community Events',
      description: 'The app promotes community engagement by hosting and advertising events focused on mental health awareness and well-being.',
    },
  ];

  return (
    <div className="py-24 sm:py-32" style={{ backgroundColor: "#0d87f430" }}>
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl" style={{ textAlign: "center" }}>Features</h2>
        <div className="mt-2 text-lg leading-8 text-gray-600" style={{ textAlign: "center" }}>
          Empower Your Well-Being.
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mt-10">
          {features.map((feature, index) => (
            <div key={index} className="p-8 space-y-3 border-2 border-blue-400 rounded-xl">
              <h2 className="text-2xl font-semibold text-gray-800 capitalize">
                {feature.title}
              </h2>
              <p className="text-gray-700">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Features;
