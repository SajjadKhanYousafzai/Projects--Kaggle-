import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Blood Cell Anomaly Detection | AI-Powered Hematology Dashboard",
  description:
    "Interactive dashboard for blood cell anomaly detection using machine learning. Visualize model performance, cell type distributions, and clinical insights.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
