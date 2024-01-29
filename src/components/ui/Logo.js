import React from "react";
import { Box, Image } from "@chakra-ui/react";

export default function Logo(props) {
  return (
    <Box {...props}>
      <Image src="/CommunityLogo.png" alt="Community Logo" w={48} h={24} />
    </Box>
  );
}
