param(
    [Parameter(Mandatory=$true)][string]$PptxPath,
    [Parameter(Mandatory=$true)][string]$PreviewDirectory,
    [Parameter(Mandatory=$true)][string]$ManifestPath
)

$ErrorActionPreference = 'Stop'
$msoTrue = -1
$msoFalse = 0
$msoTextOrientationHorizontal = 1
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoShapeOval = 9
$msoShape5pointStar = 92
$msoConnectorStraight = 1
$msoArrowheadTriangle = 3
$msoBringToFront = 0
$ppLayoutBlank = 12

function Rgb([int]$r, [int]$g, [int]$b) { return [int]($r + 256*$g + 65536*$b) }

$C = @{
    Ink      = Rgb 38 50 56
    Mid      = Rgb 102 114 122
    Light    = Rgb 217 224 228
    Panel    = Rgb 247 249 250
    H1       = Rgb 76 120 168
    H2       = Rgb 231 111 81
    Topology = Rgb 42 157 143
    Example  = Rgb 108 142 191
    Rule     = Rgb 155 106 166
    PaleTeal = Rgb 239 248 246
    PaleRose = Rgb 255 249 246
    PaleLilac= Rgb 251 250 252
    White    = Rgb 255 255 255
}

$script:shapeCounter = 0
function NextName([string]$prefix) {
    $script:shapeCounter++
    return ('{0}_{1:D3}' -f $prefix, $script:shapeCounter)
}

function SetShapeName($shape, [string]$name) {
    $shape.Name = $name
    return $shape
}

function AddBox($slide, [float]$x, [float]$y, [float]$w, [float]$h,
                [int]$fill, [int]$line, [float]$weight, [string]$name) {
    $s = $slide.Shapes.AddShape($msoShapeRoundedRectangle, $x, $y, $w, $h)
    $s.Fill.Visible = $msoTrue
    $s.Fill.Solid()
    $s.Fill.ForeColor.RGB = $fill
    $s.Line.Visible = $msoTrue
    $s.Line.ForeColor.RGB = $line
    $s.Line.Weight = $weight
    $s.Adjustments.Item(1) = 0.12
    return SetShapeName $s $name
}

function AddText($slide, [float]$x, [float]$y, [float]$w, [float]$h,
                 [string]$text, [float]$size, [int]$color, [bool]$bold,
                 [int]$align, [string]$font, [string]$name) {
    $s = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $x, $y, $w, $h)
    $s.Fill.Visible = $msoFalse
    $s.Line.Visible = $msoFalse
    $s.TextFrame2.MarginLeft = 2
    $s.TextFrame2.MarginRight = 2
    $s.TextFrame2.MarginTop = 1
    $s.TextFrame2.MarginBottom = 1
    $s.TextFrame2.VerticalAnchor = 3
    $s.TextFrame2.TextRange.Text = $text
    $s.TextFrame2.TextRange.Font.Name = $font
    $s.TextFrame2.TextRange.Font.Size = $size
    $s.TextFrame2.TextRange.Font.Bold = $(if($bold){$msoTrue}else{$msoFalse})
    $s.TextFrame2.TextRange.Font.Fill.ForeColor.RGB = $color
    $s.TextFrame2.TextRange.ParagraphFormat.Alignment = $align
    $s.TextFrame2.TextRange.ParagraphFormat.SpaceAfter = 0
    return SetShapeName $s $name
}

function AddLine($slide, [float]$x1, [float]$y1, [float]$x2, [float]$y2,
                 [int]$color, [float]$weight, [bool]$arrow, [bool]$dash,
                 [string]$name) {
    $s = $slide.Shapes.AddConnector($msoConnectorStraight, $x1, $y1, $x2, $y2)
    $s.Line.ForeColor.RGB = $color
    $s.Line.Weight = $weight
    if ($arrow) { $s.Line.EndArrowheadStyle = $msoArrowheadTriangle }
    if ($dash) { $s.Line.DashStyle = 4 }
    $s.ZOrder($msoBringToFront)
    return SetShapeName $s $name
}

function AddCircle($slide, [float]$cx, [float]$cy, [float]$diameter,
                   [int]$fill, [int]$line, [float]$weight, [string]$name) {
    $s = $slide.Shapes.AddShape($msoShapeOval, $cx-$diameter/2, $cy-$diameter/2, $diameter, $diameter)
    $s.Fill.Solid(); $s.Fill.ForeColor.RGB = $fill
    $s.Line.ForeColor.RGB = $line; $s.Line.Weight = $weight
    return SetShapeName $s $name
}

function AddStar($slide, [float]$cx, [float]$cy, [float]$diameter,
                 [int]$fill, [string]$name) {
    $s = $slide.Shapes.AddShape($msoShape5pointStar, $cx-$diameter/2, $cy-$diameter/2, $diameter, $diameter)
    $s.Fill.Solid(); $s.Fill.ForeColor.RGB = $fill
    $s.Line.Visible = $msoFalse
    return SetShapeName $s $name
}

function AddInputIcon($slide) {
    $points = @(@(47,151), @(80,128), @(112,145), @(145,130))
    for($i=0; $i -lt $points.Count-1; $i++) {
        AddLine $slide $points[$i][0] $points[$i][1] $points[$i+1][0] $points[$i+1][1] $C.H1 3 $false $false (NextName 'input_network_line') | Out-Null
    }
    foreach($p in $points[0..2]) { AddCircle $slide $p[0] $p[1] 13 $C.White $C.H1 1.4 (NextName 'input_network_node') | Out-Null }
    AddStar $slide 145 130 15 $C.H2 (NextName 'input_source') | Out-Null
    foreach($x in @(50,103)) {
        $tag = AddBox $slide $x 178 42 27 $C.White (Rgb 185 194 199) 1 (NextName 'missing_tag')
        $tag.Line.DashStyle = 4
        AddText $slide $x 178 42 27 '?' 11 $C.Mid $false 2 'Arial' (NextName 'missing_mark') | Out-Null
    }
    AddText $slide 30 228 125 48 "Incomplete OSM`npower facilities" 12.5 $C.Ink $true 2 'Arial' (NextName 'input_label') | Out-Null
}

function AddOutputSchematic($slide) {
    AddText $slide 810 105 125 34 "Role mapping`nand graph assembly" 11.5 $C.Ink $true 2 'Arial' (NextName 'assembly_heading') | Out-Null

    AddLine $slide 818 166 846 166 $C.H1 3 $false $false (NextName 'role_line') | Out-Null
    AddCircle $slide 832 193 14 $C.White $C.Ink 1.3 (NextName 'role_node') | Out-Null
    AddStar $slide 832 222 18 $C.H2 (NextName 'role_source') | Out-Null
    AddLine $slide 851 195 865 195 $C.Mid 1.4 $true $false (NextName 'role_to_graph') | Out-Null

    AddCircle $slide 875 195 16 $C.White $C.H2 1.4 (NextName 'generator') | Out-Null
    AddText $slide 868 188 14 14 'G' 6.5 $C.H2 $true 2 'Arial' (NextName 'generator_label') | Out-Null
    AddLine $slide 883 195 889 195 $C.Topology 1.6 $false $false (NextName 'generator_line') | Out-Null
    AddCircle $slide 895 195 12 $C.White $C.Topology 1.2 (NextName 'transformer_primary') | Out-Null
    AddCircle $slide 904 195 12 $C.White $C.Topology 1.2 (NextName 'transformer_secondary') | Out-Null
    AddLine $slide 910 195 918 195 $C.Topology 1.6 $false $false (NextName 'transformer_line') | Out-Null
    AddLine $slide 918 168 918 222 $C.Ink 3 $false $false (NextName 'busbar') | Out-Null
    foreach($entry in @(@(936,174),@(938,195),@(936,216))) {
        AddLine $slide 918 195 $entry[0] $entry[1] $C.Topology 1.4 $false $false (NextName 'feeder') | Out-Null
        AddCircle $slide $entry[0] $entry[1] 9 $C.White $C.Topology 1.1 (NextName 'feeder_node') | Out-Null
    }
}

function AddCard($slide, [float]$x, [float]$y, [float]$w, [float]$h,
                 [int]$accent, [string]$heading, [string]$detail, [string]$font, [string]$prefix) {
    AddBox $slide $x $y $w $h $C.Panel $C.Light 1 (NextName ($prefix+'_box')) | Out-Null
    $bar = $slide.Shapes.AddShape($msoShapeRectangle, $x+7, $y+7, 5, $h-14)
    $bar.Fill.Solid(); $bar.Fill.ForeColor.RGB = $accent; $bar.Line.Visible = $msoFalse
    SetShapeName $bar (NextName ($prefix+'_accent')) | Out-Null
    AddText $slide ($x+24) ($y+14) ($w-32) 22 $heading 11.2 $C.Ink $true 1 $font (NextName ($prefix+'_heading')) | Out-Null
    AddText $slide ($x+24) ($y+43) ($w-32) 30 $detail 8.7 $C.Mid $false 1 $font (NextName ($prefix+'_detail')) | Out-Null
}

function BuildSlide($slide) {
    $font = 'Arial'
    $degree = [char]176
    $slide.FollowMasterBackground = $msoFalse
    $slide.Background.Fill.Solid(); $slide.Background.Fill.ForeColor.RGB = $C.White

    # Containers
    AddBox $slide 20 95 145 210 $C.Panel $C.Topology 1.7 (NextName 'input_box') | Out-Null
    AddBox $slide 190 125 150 150 $C.PaleTeal $C.Topology 1.7 (NextName 'visible_input_box') | Out-Null
    AddBox $slide 445 90 120 75 $C.White $C.H1 2.0 (NextName 'h1_box') | Out-Null
    AddBox $slide 445 210 120 75 $C.White $C.H2 2.0 (NextName 'h2_box') | Out-Null
    AddBox $slide 380 305 250 85 $C.PaleLilac (Rgb 198 181 204) 1.4 (NextName 'evidence_box') | Out-Null
    AddBox $slide 650 125 125 150 $C.Panel $C.Ink 1.7 (NextName 'prediction_box') | Out-Null
    AddBox $slide 800 95 145 210 $C.PaleRose $C.H2 1.7 (NextName 'assembly_box') | Out-Null

    AddInputIcon $slide

    AddText $slide 200 160 130 80 "Masked class label`nVisible OSM attributes`nShared topology context" 10.8 $C.Ink $false 2 $font (NextName 'visible_input_text') | Out-Null
    AddText $slide 457 105 96 45 "H1`nLLM" 13 $C.H1 $true 2 'Arial' (NextName 'h1_text') | Out-Null
    AddText $slide 457 225 96 45 "H2`nRAG-LLM" 13 $C.H2 $true 2 'Arial' (NextName 'h2_text') | Out-Null

    AddText $slide 390 315 230 20 'Retrieved evidence for H2' 10.5 $C.Ink $true 2 $font (NextName 'evidence_title') | Out-Null
    AddLine $slide 505 340 505 378 (Rgb 221 212 225) 1 $false $false (NextName 'evidence_divider') | Out-Null
    AddText $slide 390 340 110 35 "Labelled examples`n(BM25)" 8.3 $C.Example $false 2 $font (NextName 'example_evidence') | Out-Null
    AddText $slide 510 340 110 35 "Power system rules`n(exact match)" 8.3 $C.Rule $false 2 $font (NextName 'rule_evidence') | Out-Null

    AddText $slide 660 163 105 75 "Facility`ncategory`nprediction" 12 $C.Ink $true 2 $font (NextName 'prediction_text') | Out-Null
    AddOutputSchematic $slide

    # Main flow arrows; all terminate at the midpoint of a box edge.
    AddLine $slide 165 200 190 200 $C.Mid 1.8 $true $false (NextName 'flow_input_to_visible') | Out-Null
    AddLine $slide 340 200 392 200 $C.Topology 1.8 $false $false (NextName 'flow_split_trunk') | Out-Null
    AddLine $slide 392 127.5 392 247.5 $C.Topology 1.8 $false $false (NextName 'flow_split_vertical') | Out-Null
    AddLine $slide 392 127.5 445 127.5 $C.Topology 1.8 $true $false (NextName 'flow_to_h1') | Out-Null
    AddLine $slide 392 247.5 445 247.5 $C.Topology 1.8 $true $false (NextName 'flow_to_h2') | Out-Null
    AddLine $slide 505 305 505 285 $C.Rule 1.8 $true $false (NextName 'evidence_to_h2') | Out-Null
    AddLine $slide 565 127.5 650 200 $C.H1 1.8 $true $false (NextName 'h1_to_prediction') | Out-Null
    AddLine $slide 565 247.5 650 200 $C.H2 1.8 $true $false (NextName 'h2_to_prediction') | Out-Null
    AddLine $slide 775 200 800 200 $C.Mid 1.8 $true $false (NextName 'prediction_to_assembly') | Out-Null

    # Lower aligned design checks; guides have no arrowheads and do not imply sequence.
    AddLine $slide 25 408 935 408 $C.Light 1 $false $false (NextName 'lower_separator') | Out-Null
    foreach($x in @(245,505,795)) { AddLine $slide $x 408 $x 430 $C.Light 1 $false $true (NextName 'alignment_guide') | Out-Null }
    AddCard $slide 130 430 230 90 $C.Topology 'Spatial isolation' ("0.125{0} blocks; five folds" -f $degree) $font 'spatial_card'
    AddCard $slide 390 430 230 90 $C.H1 'Paired evaluation' 'same objects; paired calls; three repeats' $font 'paired_card'
    AddCard $slide 650 430 290 90 $C.H2 'Prespecified evaluation' 'category recovery; role mapping; graph assembly' $font 'outcome_card'
}

New-Item -ItemType Directory -Path $PreviewDirectory -Force | Out-Null
$createdApp = $false
try {
    try { $app = [Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application') }
    catch { $app = New-Object -ComObject PowerPoint.Application; $createdApp = $true }
    $app.Visible = $msoTrue
    $pres = $app.Presentations.Open($PptxPath, $msoFalse, $msoFalse, $msoFalse)
    $pres.PageSetup.SlideWidth = 960
    $pres.PageSetup.SlideHeight = 540
    while($pres.Slides.Count -gt 0) { $pres.Slides.Item(1).Delete() }

    $slideEn = $pres.Slides.Add(1, $ppLayoutBlank)
    $slideEn.Name = 'Figure1_English_Editable'
    BuildSlide $slideEn

    $pres.Save()
    $slideEn.Export((Join-Path $PreviewDirectory 'Figure1_English_Editable.png'), 'PNG', 1920, 1080)
    $manifest = @()
    foreach($slide in @($slideEn)) {
        $typeCounts = @{}
        foreach($shape in @($slide.Shapes)) {
            $key = [string]$shape.Type
            if(-not $typeCounts.ContainsKey($key)){ $typeCounts[$key]=0 }
            $typeCounts[$key]++
        }
        $manifest += [pscustomobject]@{
            slide_index = $slide.SlideIndex
            slide_name = $slide.Name
            shape_count = $slide.Shapes.Count
            shape_type_counts = $typeCounts
            pictures_or_linked_pictures = @($slide.Shapes | Where-Object { $_.Type -in @(11,13) }).Count
        }
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8
    $pres.Close()
} finally {
    if($pres) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($pres) }
    if($app) {
        if($createdApp) { $app.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app)
    }
    [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}

Write-Output "PPTX=$PptxPath"
Write-Output "PREVIEWS=$PreviewDirectory"
Write-Output "MANIFEST=$ManifestPath"
