import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { Auth0Provider, useAuth0 } from "@auth0/auth0-react";
import Landing from "./pages/Landing";
import Contact from "./pages/Contact";
import Features from "./pages/Features";
import ShareStories from "./pages/ShareStories";
import Home from "./pages/Home";
import Volunteer from "./pages/Volunteer";
import CommunityEvents from "./pages/CommunityEvents";
import Videos from "./pages/Videos";


///-----------------------------------------
import JournalSection from "./pages/JournalSection"

///-------------------------

const auth0Domain = "dev-8io25ngx3xtfx5n2.us.auth0.com"
const auth0ClientId = "Simmznku4ZKYPP0lo1gaEXEB0QkiQPmu"

export default function App() {
  function AuthenticatedRoute({ element }) {
    const { isAuthenticated } = useAuth0()
    return isAuthenticated ? element : <Navigate to="/" />
  }

  return (
    <Auth0Provider
      domain={auth0Domain}
      clientId={auth0ClientId}
      redirectUri={window.location.origin}
    >
      <Router>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/features" element={<Features />} />
          <Route path="/ShareStories" element={<ShareStories />} />
          <Route path="/journal" element={<JournalSection />} />
          <Route
            path="/Home"
            element={<AuthenticatedRoute element={<Home />} />}
          />
          <Route
            path="/volunteer"
            element={<AuthenticatedRoute element={<Volunteer />} />}
          />
          <Route
            path="/CommunityEvents"
            element={<AuthenticatedRoute element={<CommunityEvents />} />}
          />
          <Route
            path="/videos"
            element={<AuthenticatedRoute element={<Videos />} />}
          />
        </Routes>
      </Router>
    </Auth0Provider>
  )
}
