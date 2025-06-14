#include <Python.h>
#include <WinSock2.h>
#include <deque>
#include <thread>
#include <mutex>
#include <cmath>
#include <cstdio>

#pragma comment(lib, "WS2_32.lib")

typedef struct {
    float x, y, z;
    uint8_t i;
    uint32_t timestamp;
    uint8_t channel;
    float r;
    float theta;
} LiDarPoint;

static std::deque<LiDarPoint> scanPoints;
static std::mutex dataMutex;
const size_t MAX_POINTS = 100000;

static float ChList[16] = { -15.0f, 1.0f, -13.0f, 3.0f, -11.0f, 5.0f, -9.0f, 7.0f,
                            -7.0f, 9.0f, -5.0f, 11.0f, -3.0f, 13.0f, -1.0f, 15.0f };
static float offsetV[16] = { 5.06f, -9.15f, 5.06f, -9.15f, 5.06f, -9.15f, 5.06f, -9.15f,
                             9.15f, -5.06f, 9.15f, -5.06f, 9.15f, -5.06f, 9.15f, -5.06f };
static float offsetH[16] = { 21.0f, 21.0f, 21.0f, 21.0f, 21.0f, 21.0f, 21.0f, 21.0f,
                            -21.0f, -21.0f, -21.0f, -21.0f, -21.0f, -21.0f, -21.0f, -21.0f };

#pragma pack(push, 1)
typedef struct {
    uint16_t distance;
    uint8_t intensity;
} DataPoint;

typedef struct {
    uint8_t identifier[2];
    uint16_t azimuth;
    DataPoint points[32];
} DataBlock;

typedef struct {
    DataBlock blocks[12];
    uint32_t timestamp;
    uint8_t factory[2];
} RawPacket;
#pragma pack(pop)

static void AddPoint(float r, float ang, uint8_t intensity, uint8_t chIndex, uint32_t timestamp) {
    float _ang = (ang + chIndex * 0.0108f);
    float _ang_rad = _ang * 3.14159265f / 180.0f;
    float _w = ChList[chIndex] * 3.14159265f / 180.0f;

    float x = r * cos(_w) * cos(_ang_rad) + offsetH[chIndex] * cos(_ang_rad) * 0.001f;
    float y = r * cos(_w) * sin(_ang_rad) - offsetH[chIndex] * sin(_ang_rad) * 0.001f;
    float z = r * sin(_w) + offsetV[chIndex] * 0.001f;

    std::lock_guard<std::mutex> lock(dataMutex);
    if (scanPoints.size() >= MAX_POINTS) {
        scanPoints.pop_front();  // 环形缓冲策略：移除最老点
    }
    scanPoints.push_back({x, y, z, intensity, timestamp, chIndex, r, _ang});
}

static void ReceiveThread() {
    printf("[INFO] ReceiveThread started.\n");

    WSADATA wsd;
    if (WSAStartup(MAKEWORD(2, 2), &wsd) != 0) {
        printf("[ERROR] WSAStartup failed: %d\n", WSAGetLastError());
        return;
    }

    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) {
        printf("[ERROR] socket() failed: %d\n", WSAGetLastError());
        WSACleanup();
        return;
    }

    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    SOCKADDR_IN sRecvAddr = {};
    sRecvAddr.sin_family = AF_INET;
    sRecvAddr.sin_port = htons(2368);
    sRecvAddr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);

    if (bind(s, (SOCKADDR*)&sRecvAddr, sizeof(sRecvAddr)) != 0) {
        printf("[ERROR] bind() failed: %d\n", WSAGetLastError());
        closesocket(s);
        WSACleanup();
        return;
    }

    printf("[INFO] UDP socket ready. Listening on port 2368...\n");

    SOCKADDR_IN sSendAddr;
    int nSenderAddrSize = sizeof(sSendAddr);
    char szBuf[1206] = {0};

    while (true) {
        int nResult = recvfrom(s, szBuf, sizeof(szBuf), 0, (SOCKADDR*)&sSendAddr, &nSenderAddrSize);
        if (nResult == SOCKET_ERROR) {
            printf("[WARNING] recvfrom() failed: %d\n", WSAGetLastError());
            continue;
        }

        if (nResult != 1206) {
            printf("[WARNING] Unexpected packet size: %d bytes\n", nResult);
            continue;
        }

        RawPacket* packet = (RawPacket*)szBuf;
        uint32_t timestamp = packet->timestamp;

        for (int i = 0; i < 12; ++i) {
            float angle = packet->blocks[i].azimuth * 0.01f;
            for (int j = 0; j < 16; ++j) {
                float r = packet->blocks[i].points[j].distance * 0.002f;
                uint8_t intensity = packet->blocks[i].points[j].intensity;
                AddPoint(r, angle, intensity, j, timestamp);
            }
            for (int j = 0; j < 16; ++j) {
                float r = packet->blocks[i].points[j + 16].distance * 0.002f;
                uint8_t intensity = packet->blocks[i].points[j + 16].intensity;
                AddPoint(r, angle + 0.18f, intensity, j, timestamp);
            }
        }
    }

    closesocket(s);
    WSACleanup();
}

static PyObject* get_latest_points(PyObject* self, PyObject* args) {
    std::lock_guard<std::mutex> lock(dataMutex);
    PyObject* list = PyList_New(scanPoints.size());
    for (size_t i = 0; i < scanPoints.size(); ++i) {
        const LiDarPoint& p = scanPoints[i];
        PyObject* point = Py_BuildValue("(fffBIBff)", p.x, p.y, p.z, p.i, p.timestamp, p.channel, p.r, p.theta);
        PyList_SetItem(list, i, point);
    }
    scanPoints.clear();  // 可选，是否清空视需求决定
    return list;
}

static PyMethodDef LidarMethods[] = {
    {"get_latest_points", get_latest_points, METH_NOARGS, "Get latest LiDAR points"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef lidarmodule = {
    PyModuleDef_HEAD_INIT, "pylidar", NULL, -1, LidarMethods
};

PyMODINIT_FUNC PyInit_pylidar(void) {
    std::thread(ReceiveThread).detach();
    return PyModule_Create(&lidarmodule);
}
