import React from "react";
import { Link } from "react-router-dom";
import { Box, Flex, Text, Button } from "@chakra-ui/react";
import Logo from "../ui/Logo";
import { useAuth0 } from "@auth0/auth0-react";

const MenuItem = ({ children, isLast, to = "/", ...rest }) => {
  return (
    <Text
      mb={{ base: isLast ? 0 : 8, sm: 0 }}
      mr={{ base: 0, sm: isLast ? 0 : 8 }}
      display="block"
      {...rest}
    >
      <Link to={to}>{children}</Link>
    </Text>
  );
};

const CloseIcon = () => (
  <svg width="24" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
    {/* ... */}
  </svg>
);

const MenuIcon = () => (
  <svg
    width="24px"
    viewBox="0 0 20 20"
    xmlns="http://www.w3.org/2000/svg"
    fill="white"
  >
    {/* ... */}
  </svg>
);

const AuthenticatedHeader = (props) => {
  const [show, setShow] = React.useState(false);
  const toggleMenu = () => setShow(!show);

  const { logout } = useAuth0();

  return (
    <Flex
      as="nav"
      align="center"
      justify="space-between"
      wrap="wrap"
      w="100%"
      mb={8}
      p={8}
      bg={["primary.500", "primary.500", "transparent", "transparent"]}
      color={["white", "white", "primary.700", "primary.700"]}
      {...props}
    >
      <Flex align="center">
        <Logo
          w="100px"
          color={["white", "white", "primary.500", "primary.500"]}
        />
      </Flex>

      <Box display={{ base: "block", md: "none" }} onClick={toggleMenu}>
        {show ? <CloseIcon /> : <MenuIcon />}
      </Box>

      <Box
        display={{ base: show ? "block" : "none", md: "block" }}
        flexBasis={{ base: "100%", md: "auto" }}
      >
        <Flex
          align="center"
          justify={["center", "space-between", "flex-end", "flex-end"]}
          direction={["column", "row", "row", "row"]}
          pt={[4, 4, 0, 0]}
        >
          <MenuItem to="/Home">Home </MenuItem>
          <MenuItem to="/CommunityEvents">Community Events </MenuItem>
          <MenuItem to="/journal">Personal Journal</MenuItem>
          <MenuItem to="/videos">Educational Videos</MenuItem>
          <MenuItem to="/volunteer">Volunteering</MenuItem>
          <Button onClick={() => logout({ returnTo: window.location.origin })}>
            Logout
          </Button>
        </Flex>
      </Box>
    </Flex>
  );
};

export default AuthenticatedHeader;
