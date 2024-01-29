import React from "react";
import Hero from "../components/sections/Hero";
import LandingLayout from "../components/layouts/LandingLayout";
import { useAuth0 } from "@auth0/auth0-react";
import { useNavigate } from "react-router-dom";

export default function Landing() {
 const { isAuthenticated, loginWithPopup } = useAuth0();
 const navigate = useNavigate();

 const handleLogin = () => {
   loginWithPopup();
 };

 if (isAuthenticated) {
   navigate("/Home");
 }

  return (
    <LandingLayout>
      <Hero
        title="Mental Milestone"
        subtitle="Take the first steps toward improved mental health with us today. 'Empower Your Well-Being'."
        image="/Hero.jpg"
        ctaText="Login/Register"
        handleLogin={handleLogin}
      />
    </LandingLayout>
  );
}
