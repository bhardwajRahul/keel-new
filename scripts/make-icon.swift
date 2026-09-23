import AppKit
import Foundation

let destination = CommandLine.arguments[1]
let side = 1024
let image = NSImage(size: NSSize(width: side, height: side))
image.lockFocus()

let background = NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: side, height: side), xRadius: 230, yRadius: 230)
NSColor(calibratedRed: 0.075, green: 0.09, blue: 0.115, alpha: 1).setFill()
background.fill()

let glow = NSBezierPath(ovalIn: NSRect(x: 190, y: 160, width: 644, height: 644))
NSColor(calibratedRed: 0.10, green: 0.70, blue: 0.62, alpha: 0.10).setFill()
glow.fill()

let mark = NSBezierPath()
mark.lineWidth = 52
mark.lineCapStyle = .round
mark.lineJoinStyle = .round
mark.move(to: NSPoint(x: 270, y: 735))
mark.line(to: NSPoint(x: 270, y: 276))
mark.curve(to: NSPoint(x: 512, y: 208), controlPoint1: NSPoint(x: 335, y: 226), controlPoint2: NSPoint(x: 430, y: 208))
mark.curve(to: NSPoint(x: 754, y: 276), controlPoint1: NSPoint(x: 594, y: 208), controlPoint2: NSPoint(x: 689, y: 226))
mark.line(to: NSPoint(x: 754, y: 735))
NSColor(calibratedRed: 0.36, green: 0.89, blue: 0.76, alpha: 1).setStroke()
mark.stroke()

let route = NSBezierPath()
route.lineWidth = 52
route.lineCapStyle = .round
route.lineJoinStyle = .round
route.move(to: NSPoint(x: 270, y: 655))
route.line(to: NSPoint(x: 512, y: 414))
route.line(to: NSPoint(x: 754, y: 655))
route.move(to: NSPoint(x: 512, y: 414))
route.line(to: NSPoint(x: 512, y: 208))
route.stroke()

image.unlockFocus()
guard let tiff = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiff),
      let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Could not render icon")
}
try png.write(to: URL(fileURLWithPath: destination))
