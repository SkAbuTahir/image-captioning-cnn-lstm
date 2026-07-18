import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Image Captioning — ResNet50 + LSTM",
  description: "Automatic image captioning using ResNet50 encoder and LSTM decoder, trained on Flickr8k.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
