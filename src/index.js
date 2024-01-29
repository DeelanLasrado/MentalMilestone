import React from "react"
import ReactDOM from "react-dom"
import { ChakraProvider } from "@chakra-ui/react"
import App from "./App"
import customTheme from "./utils/theme"
import "./index.css"
import { Provider } from "./context/AppContext"

const rootElement = document.getElementById("root")

ReactDOM.render(
  <ChakraProvider theme={customTheme}>
    <Provider>
      <App />
    </Provider>
  </ChakraProvider>,
  rootElement
)
