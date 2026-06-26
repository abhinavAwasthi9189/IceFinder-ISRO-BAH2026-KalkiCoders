#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <algorithm>
#include <set>
#include <map>
#include <string>

// --------------- data structures ------------------------------

struct BinaryHeader {
    uint32_t width;
    uint32_t height;
    double   minX; 
    double   maxX; 
    double   minY; 
    double   maxY; 
};

struct PSRNode {
    float weight;    // 1.0=normal, 0.5=PSR, 0.1=doubly shadowed
    float elevation;
};

struct GeoBounds {
    double minX, maxX, minY, maxY;
    uint32_t width, height;

    void pixelToWorld(uint32_t row, uint32_t col, double& worldX, double& worldY) const {
        double cellWidth = std::abs(maxX - minX) / width;
        double cellHeight = std::abs(maxY - minY) / height;
        worldX = std::min(minX, maxX) + col * cellWidth;
        worldY = std::max(minY, maxY) - row * cellHeight;
    }

    bool worldToPixel(double worldX, double worldY, uint32_t& row, uint32_t& col) const {
        double rMinX = std::min(minX, maxX), rMaxX = std::max(minX, maxX);
        double rMinY = std::min(minY, maxY), rMaxY = std::max(minY, maxY);
        if (worldX < rMinX || worldX > rMaxX || worldY < rMinY || worldY > rMaxY) return false;
        double cw = (rMaxX - rMinX) / width;
        double ch = (rMaxY - rMinY) / height;
        col = static_cast<uint32_t>((worldX - rMinX) / cw);
        row = static_cast<uint32_t>((rMaxY - worldY) / ch);
        col = std::min(col, width  - 1);
        row = std::min(row, height - 1);
        return true;
    }
};

struct IceTarget {
    double   worldX, worldY;
    uint32_t psrRow, psrCol;
    uint32_t dfsarRow, dfsarCol;
    float    psrWeight;
    float    elevation;
    float    iceFraction; 
};

#pragma pack(push, 1)
struct PSRLocationRecord {
    double   centerLat;
    double   centerLon;
    uint32_t pixelCount;     
    uint32_t icePxCount;     
    float    iceVolMid;
    float    iceVolMin;
    float    iceVolMax;
    float    iceFrac;        
    uint8_t  category;       
};
#pragma pack(pop)

struct RegionAccumulator {
    uint64_t sumRow = 0, sumCol = 0;   
    uint32_t pixelCount = 0;
    uint32_t icePxCount = 0;
    float    totalIceFraction = 0.0f;
    uint8_t  category = 0;             
    uint32_t optRow = 0;               
    uint32_t optCol = 0;               
};

// ---------------main pipeline class---------------------

class IceSensorFusionStreamer {
private:
    GeoBounds   psrBounds, dfsarBounds;
    std::string psrFilePath;

    static constexpr double MOON_RADIUS_M = 1737400.0;
    static constexpr double DEG_TO_M      = M_PI / 180.0 * MOON_RADIUS_M;

    static constexpr float ICE_DEPTH_M   = 5.0f;
    static constexpr float PSR_PIXEL_M   = 118.5f;

    //require at least 2 connected pixels to ignore scattered noise
    static constexpr uint32_t MIN_ICE_PIXELS_PER_REGION = 2; 

    //massive increase to volume requirement (100,000 m3)
    static constexpr float MIN_ICE_VOLUME_M3 = 100000.0f; 

    //hard cap on how many targets are sent to the python lander
    static constexpr size_t MAX_TARGETS_TO_EXPORT = 10;

    static constexpr size_t HEADER_OFFSET = sizeof(BinaryHeader) < 40 ? 40 : sizeof(BinaryHeader);

    bool readHeader(const std::string& path, GeoBounds& b, bool isDFSAR) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        
        BinaryHeader header;
        f.read(reinterpret_cast<char*>(&header.width),  4);
        f.read(reinterpret_cast<char*>(&header.height), 4);
        f.read(reinterpret_cast<char*>(&header.minX),   8);
        f.read(reinterpret_cast<char*>(&header.maxX),   8);
        f.read(reinterpret_cast<char*>(&header.minY),   8);
        f.read(reinterpret_cast<char*>(&header.maxY),   8);
        
        b.width = header.width;
        b.height = header.height;
        b.minX = header.minX;
        b.maxX = header.maxX;
        b.minY = header.minY;
        b.maxY = header.maxY;

        std::cout << (isDFSAR ? "DFSAR" : "PSR  ") << " Map: "
                  << b.width << "x" << b.height << "\n"
                  << "   Bounds: X[" << b.minX << " to " << b.maxX
                  << "] Y[" << b.minY << " to " << b.maxY << "]\n";
        return true;
    }

    void worldToSelenoGraphic(double wx, double wy, double& lat, double& lon) const {
        lon = wx / DEG_TO_M;
        lat = wy / DEG_TO_M;
    }

    // MAPS BOTH TIERED CATEGORIES (0,1,2) AND EXACT FRACTIONS (0.0 - 0.5)
    bool buildIceMaskAndCategoryGrid(const std::vector<IceTarget>& allIce, 
                                     std::vector<uint8_t>& iceMaskOut, 
                                     std::vector<float>& iceFracOut,
                                     std::vector<uint8_t>& categoryOut) {
        uint32_t W = psrBounds.width, H = psrBounds.height;
        size_t   N = static_cast<size_t>(W) * H;
        iceMaskOut.assign(N, 0);
        iceFracOut.assign(N, 0.0f);
        categoryOut.assign(N, 0);

        std::ifstream psrFile(psrFilePath, std::ios::binary);
        if (!psrFile) {
            std::cerr << "Error: cannot reopen PSR file for category grid\n";
            return false;
        }
        psrFile.seekg(HEADER_OFFSET, std::ios::beg);

        std::vector<PSRNode> rowBuf(W);
        for (uint32_t r = 0; r < H; ++r) {
            psrFile.read(reinterpret_cast<char*>(rowBuf.data()), W * sizeof(PSRNode));
            if (!psrFile) break;
            
            for (uint32_t c = 0; c < W; ++c) {
                float w = rowBuf[c].weight;
                size_t idx = static_cast<size_t>(r) * W + c;
                if (w <= 0.15f)      categoryOut[idx] = 2;  // Doubly shadowed
                else if (w <= 0.9f)  categoryOut[idx] = 1;  // Regular PSR
            }
        }

        for (auto& t : allIce) {
            size_t idx = static_cast<size_t>(t.psrRow) * W + t.psrCol;
            iceFracOut[idx] = t.iceFraction; // Store true physical fraction

            if (categoryOut[idx] == 2) {
                iceMaskOut[idx] = 2; // Priority Surface Ice Trap
            } else if (categoryOut[idx] == 1) {
                iceMaskOut[idx] = 1; // Protected Subsurface Ice Trap
            }
        }
        return true;
    }

    std::vector<RegionAccumulator> findConnectedRegions(std::vector<uint8_t>& category,
                                                         const std::vector<uint8_t>& iceMask,
                                                         const std::vector<float>& iceFracGrid,
                                                         uint32_t width, uint32_t height) {
        std::vector<RegionAccumulator> regions;
        std::vector<std::pair<int,int>> stack;
        std::vector<std::pair<int,int>> componentPixels; 

        for (uint32_t r = 0; r < height; ++r) {
            for (uint32_t c = 0; c < width; ++c) {
                size_t idx = static_cast<size_t>(r) * width + c;
                uint8_t cat = category[idx];
                if (cat == 0) continue;

                RegionAccumulator acc;
                acc.category = cat;
                stack.clear();
                componentPixels.clear();

                stack.push_back(std::make_pair(static_cast<int>(r), static_cast<int>(c)));
                category[idx] = 0; 

                while (!stack.empty()) {
                    std::pair<int,int> cur = stack.back();
                    int cr = cur.first;
                    int cc = cur.second;
                    stack.pop_back();

                    componentPixels.push_back({cr, cc});

                    size_t cidx = static_cast<size_t>(cr) * width + cc;
                    acc.pixelCount++;
                    acc.sumRow += cr;
                    acc.sumCol += cc;
                    
                    if (iceMask[cidx] > 0) {
                        acc.icePxCount++;
                        acc.totalIceFraction += iceFracGrid[cidx]; // Exact volumetric sum
                    }

                    for (int dr = -1; dr <= 1; ++dr) {
                        for (int dc = -1; dc <= 1; ++dc) {
                            if (dr == 0 && dc == 0) continue;
                            int nr = cr + dr, nc = dc + cc;
                            if (nr < 0 || nr >= static_cast<int>(height) ||
                                nc < 0 || nc >= static_cast<int>(width)) continue;
                            size_t nidx = static_cast<size_t>(nr) * width + nc;
                            if (category[nidx] == cat) {
                                category[nidx] = 0;
                                stack.push_back(std::make_pair(nr, nc));
                            }
                        }
                    }
                }

                double centroidRow = static_cast<double>(acc.sumRow) / acc.pixelCount;
                double centroidCol = static_cast<double>(acc.sumCol) / acc.pixelCount;

                uint32_t bestRow = r;
                uint32_t bestCol = c;
                float maxIceWeightScore = -1.0f;
                double minDistanceToCentroid = 1e9; 

                //FRACTION-WEIGHTED TARGETING: 
                for (const auto& px : componentPixels) {
                    int cr = px.first;
                    int cc = px.second;
                    
                    if (iceMask[static_cast<size_t>(cr) * width + cc] == 0) continue;

                    float iceWeightScore = 0.0f;
                    for (int wr = -10; wr <= 10; ++wr) {
                        for (int wc = -10; wc <= 10; ++wc) {
                            int nr = cr + wr;
                            int nc = cc + wc;
                            if (nr >= 0 && nr < static_cast<int>(height) &&
                                nc >= 0 && nc < static_cast<int>(width)) {
                                size_t nidx = static_cast<size_t>(nr) * width + nc;
                                uint8_t iceType = iceMask[nidx];
                                float frac = iceFracGrid[nidx];
                                
                                if (iceType == 2) {
                                    iceWeightScore += (5.0f * frac); // 5x Priority * actual concentration
                                } else if (iceType == 1) {
                                    iceWeightScore += (1.0f * frac);
                                }
                            }
                        }
                    }

                    double dist = std::sqrt((cr - centroidRow)*(cr - centroidRow) + (cc - centroidCol)*(cc - centroidCol));

                    if (iceWeightScore > maxIceWeightScore) {
                        maxIceWeightScore = iceWeightScore;
                        minDistanceToCentroid = dist;
                        bestRow = cr;
                        bestCol = cc;
                    } 
                    else if (std::abs(iceWeightScore - maxIceWeightScore) < 0.001f) {
                        if (dist < minDistanceToCentroid) {
                            minDistanceToCentroid = dist;
                            bestRow = cr;
                            bestCol = cc;
                        }
                    }
                }

                if (maxIceWeightScore < 0.0f && !componentPixels.empty()) {
                    bestRow = componentPixels[0].first;
                    bestCol = componentPixels[0].second;
                }

                acc.optRow = bestRow;
                acc.optCol = bestCol;
                regions.push_back(acc);
            }
        }
        return regions;
    }

    void exportIceMask(const std::vector<uint8_t>& iceMask, const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Error: cannot write " << path << "\n"; return; }

        f.write(reinterpret_cast<const char*>(&psrBounds.width),  4);
        f.write(reinterpret_cast<const char*>(&psrBounds.height), 4);
        f.write(reinterpret_cast<const char*>(&psrBounds.minX), 8);
        f.write(reinterpret_cast<const char*>(&psrBounds.maxX), 8);
        f.write(reinterpret_cast<const char*>(&psrBounds.minY), 8);
        f.write(reinterpret_cast<const char*>(&psrBounds.maxY), 8);
        f.write(reinterpret_cast<const char*>(iceMask.data()), iceMask.size());

        size_t trueCount = 0;
        for (uint8_t v : iceMask) if (v > 0) trueCount++;
        std::cout << "\nGraded Ice mask exported -> " << path << "\n"
                  << "   Grid: " << psrBounds.width << "x" << psrBounds.height
                  << " | active ice pixels: " << trueCount << " / " << iceMask.size() << "\n";
    }

    void exportPSRLocations(const std::vector<RegionAccumulator>& regions, const std::string& path) {
        std::vector<PSRLocationRecord> doubly, normal;

        double cellWidth  = std::abs(psrBounds.maxX - psrBounds.minX) / psrBounds.width;
        double cellHeight = std::abs(psrBounds.maxY - psrBounds.minY) / psrBounds.height;
        double originX    = std::min(psrBounds.minX, psrBounds.maxX);
        double originY    = std::max(psrBounds.minY, psrBounds.maxY);

        float pixArea = PSR_PIXEL_M * PSR_PIXEL_M;

        for (auto& acc : regions) {
            //Drop single-pixel noise
            if (acc.icePxCount < MIN_ICE_PIXELS_PER_REGION) continue;

            float effectiveIceArea = acc.totalIceFraction * pixArea;
            float expectedVolume = effectiveIceArea * ICE_DEPTH_M;

            //Drop tiny deposits
            if (expectedVolume < MIN_ICE_VOLUME_M3) continue;

            double targetRow = static_cast<double>(acc.optRow);
            double targetCol = static_cast<double>(acc.optCol);

            double wx = originX + (targetCol + 0.5) * cellWidth;
            double wy = originY - (targetRow + 0.5) * cellHeight;

            double lat, lon;
            worldToSelenoGraphic(wx, wy, lat, lon);

            PSRLocationRecord rec;
            rec.centerLat  = lat;
            rec.centerLon  = lon;
            rec.pixelCount = acc.pixelCount;
            rec.icePxCount = acc.icePxCount;
            rec.iceVolMid  = expectedVolume; 
            rec.iceVolMin  = expectedVolume * 0.8f; 
            rec.iceVolMax  = expectedVolume * 1.2f; 
            rec.iceFrac    = acc.totalIceFraction / static_cast<float>(acc.pixelCount);
            rec.category   = (acc.category == 2) ? 1 : 0;

            if (rec.category == 1) doubly.push_back(rec);
            else                   normal.push_back(rec);
        }

        auto byVolDesc = [](const PSRLocationRecord& a, const PSRLocationRecord& b) {
            return a.iceVolMid > b.iceVolMid;
        };
        std::sort(doubly.begin(), doubly.end(), byVolDesc);
        std::sort(normal.begin(), normal.end(), byVolDesc);

        //Keep only the absolute best targets for Python
        if (doubly.size() > MAX_TARGETS_TO_EXPORT) doubly.resize(MAX_TARGETS_TO_EXPORT);
        if (normal.size() > MAX_TARGETS_TO_EXPORT) normal.resize(MAX_TARGETS_TO_EXPORT);

        std::ofstream f(path, std::ios::binary);
        if (!f) { std::cerr << "Error: cannot write " << path << "\n"; return; }

        uint32_t nDoubly = static_cast<uint32_t>(doubly.size());
        uint32_t nNormal = static_cast<uint32_t>(normal.size());
        
        f.write(reinterpret_cast<const char*>(&nDoubly), 4);
        f.write(reinterpret_cast<const char*>(&nNormal), 4);
        if (nDoubly) f.write(reinterpret_cast<const char*>(doubly.data()), doubly.size() * sizeof(PSRLocationRecord));
        if (nNormal) f.write(reinterpret_cast<const char*>(normal.data()), normal.size() * sizeof(PSRLocationRecord));

        std::cout << "\nPSR location list exported -> " << path << "\n"
                  << "   Doubly-shadowed regions: " << nDoubly << "\n"
                  << "   Normal PSR regions:      " << nNormal << "\n"
                  << "   Record size: " << sizeof(PSRLocationRecord) << " bytes each\n";

        std::cout << "\n--- Doubly-shadowed crater locations (Highest Volume & Stability first) ---\n";
        int shown = 0;
        for (auto& r : doubly) {
            if (shown++ >= 10) break;
            std::cout << "  lat=" << std::fixed << std::setprecision(3) << r.centerLat
                      << " lon=" << r.centerLon
                      << " | regionPx=" << r.pixelCount
                      << " effFrac=" << std::setprecision(2) << (r.iceFrac * 100.0f) << "%"
                      << " expectedVol=" << std::setprecision(0) << r.iceVolMid << "m3\n";
        }
    }

public:
    bool registerPSRReference(const std::string& path) {
        psrFilePath = path;
        return readHeader(path, psrBounds, false);
    }

    bool loadDFSARHeader(const std::string& path) {
        return readHeader(path, dfsarBounds, true);
    }

    void processIceIntersection(const std::string& dfsarPath, float iceFractionThreshold,
                                 const std::string& iceMaskOutputPath,
                                 const std::string& locationsOutputPath) {
        std::ifstream dfsarFile(dfsarPath, std::ios::binary);
        std::ifstream psrFile(psrFilePath,  std::ios::binary);
        if (!dfsarFile || !psrFile) {
            std::cerr << "Error: could not open input streams\n";
            return;
        }

        dfsarFile.seekg(HEADER_OFFSET, std::ios::beg);

        uint32_t procW = dfsarBounds.width;
        uint32_t procH = dfsarBounds.height;

        if (dfsarBounds.width < dfsarBounds.height) {
            std::swap(procW, procH);
        }

        std::vector<float>    row(procW);
        std::vector<IceTarget>  allIce;
        size_t highZ = 0, outOfBounds = 0;

        for (uint32_t r = 0; r < procH; ++r) {
            dfsarFile.read(reinterpret_cast<char*>(row.data()), procW * sizeof(float));
            if (!dfsarFile) break;

            for (uint32_t c = 0; c < procW; ++c) {
                float fraction = row[c];
                if (std::isnan(fraction) || std::isinf(fraction)) continue;
                
                //FILTER INCOMING RADAR DATA BY PHYSICAL ICE FRACTION(>5%)
                if (fraction < iceFractionThreshold) continue;

                ++highZ;
                double wx, wy;
                dfsarBounds.pixelToWorld(r, c, wx, wy);

                uint32_t pr, pc;
                if (!psrBounds.worldToPixel(wx, wy, pr, pc)) {
                    ++outOfBounds; continue;
                }

                size_t idx    = static_cast<size_t>(pr) * psrBounds.width + pc;
                size_t offset = HEADER_OFFSET + idx * sizeof(PSRNode);
                
                psrFile.clear(); 
                psrFile.seekg(offset, std::ios::beg);
                
                PSRNode node;
                psrFile.read(reinterpret_cast<char*>(&node), sizeof(PSRNode));

                if (!psrFile) continue;

                if (node.weight <= 0.5f && node.weight > 0.0f) {
                    allIce.push_back({wx, wy, pr, pc, r, c, node.weight, node.elevation, fraction});
                }
            }
        }

        //Sort by physical concentration
        std::sort(allIce.begin(), allIce.end(), [](const IceTarget& a, const IceTarget& b){ return a.iceFraction > b.iceFraction; });

        std::vector<IceTarget> deduped;
        std::set<std::pair<uint32_t,uint32_t>> seen;
        for (auto& t : allIce) {
            auto key = std::make_pair(t.psrRow, t.psrCol);
            if (!seen.count(key)) { seen.insert(key); deduped.push_back(t); }
        }
        allIce = deduped;

        std::cout << "\nSwath Intersection Metrics:\n"
                  << "   Above threshold (" << (iceFractionThreshold * 100.0f) << "% Ice): " << highZ << "\n"
                  << "   Outside PSR bounds: " << outOfBounds << "\n"
                  << "   Matched to PSR: " << highZ - outOfBounds << "\n"
                  << "\nTotal confirmed ice pixels (deduped): " << allIce.size() << "\n";

        std::vector<uint8_t> iceMask, category;
        std::vector<float> iceFracGrid;
        
        if (!buildIceMaskAndCategoryGrid(allIce, iceMask, iceFracGrid, category)) return;

        exportIceMask(iceMask, iceMaskOutputPath);

        auto regions = findConnectedRegions(category, iceMask, iceFracGrid, psrBounds.width, psrBounds.height);
        exportPSRLocations(regions, locationsOutputPath);
    }
};

// ----------------------- main ------------------------------------------------------------

int main() {
    IceSensorFusionStreamer pipeline;

    if (!pipeline.registerPSRReference("./lunar_psr_dem_grid.bin")) {
        std::cerr << "Failed to open PSR file\n"; return -1;
    }
    if (!pipeline.loadDFSARHeader("./lunar_grid_weights.bin")) {
        std::cerr << "Failed to open DFSAR file\n"; return -1;
    }

    pipeline.processIceIntersection(
        "./lunar_grid_weights.bin",
        0.25f, //Minimum acceptable ice fraction to track (25%)
        "./lunar_ice_mask.bin",
        "./lunar_psr_locations.bin"
    );

    return 0;
}